"""score module. Functions to score genetic interactions"""
import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings


def check_guide_input(df):
    """Check whether input DataFrame has expected format

    - Column 1: first guide identifier
    - Column 2: second guide identifier
    - Column 3: first gene identifier
    - Column 4: second gene identifier
    - Column 5+: LFC values from different conditions

    Parameters
    ----------
    df: DataFrame
        LFC data with the specified columns
    """
    if df.shape[1] < 5:
        raise ValueError('Input has ' + str(df.shape[1]) + ' columns, should be > 4')
    if df.shape[0] == 0:
        raise ValueError('Input has no rows')
    if df.iloc[:, [0, 1]].drop_duplicates().shape[0] != df.shape[0]:
        raise ValueError('The first two columns of input (guide 1 and guide 2) should uniquely identify each row')


def melt_df(df, id_cols=None, var_name='condition', value_name='lfc'):
    """Helper function to melt a DataFrame

    Parameters
    ----------
    df: DataFrame
        log fold chang dataframe
    id_cols: list, optional
        Specify id columns. If none, then the first four columns are used
    var_name: str, optional
        New variable name
    value_name: str, optional
        New value name.

    Returns
    -------
    DataFrame
    """
    if id_cols is None:
        id_cols = df.columns[0:4]  # guide1, guide2, gene1, gene2
    melted_df = df.melt(id_vars=id_cols, var_name=var_name, value_name=value_name)
    return melted_df


def order_cols(df, cols, name):
    """Reorder values in columns to be in alphabetical order

    Parameters
    ----------
    df: DataFrame
        DataFrame with at least two columns to be reodredered in alphabetical order
    cols: List
        Indices of columns to be reordered
    name: str
        Name of reordered column

    Returns
    -------
    DataFrame
        With columns in alphabetical order
    """
    col1 = df.columns[cols[0]]
    col2 = df.columns[cols[1]]
    two_col_df = df[[col1, col2]].drop_duplicates()
    two_col_df[name + '_a'] = two_col_df.apply(lambda row: (row[col1] if row[col1] <= row[col2]
                                                            else row[col2]), axis=1)
    two_col_df[name + '_b'] = two_col_df.apply(lambda row: (row[col2] if row[col1] <= row[col2]
                                                            else row[col1]), axis=1)
    ordered_df = df.merge(two_col_df, how='inner', on=[col1, col2])
    return ordered_df


def order_cols_with_meta(df, cols, meta_cols, col_name, meta_name):
    """Reorder values in columns to be in alphabetical order, keeping track of columns with
    meta-information

    Parameters
    ----------
    df: DataFrame
        DataFrame with at least two columns to be reordered in alphabetical order
    cols: List
        Indices of columns to be reordered
    meta_cols: List
        Indices of columns with meta information, ordered the same as cols
    col_name: str
        Base name of reordered columns
    meta_name: str
        Base name of meta information columns

    Returns
    -------
    DataFrame
        With columns in alphabetical order and their respective meta info
    """
    col1 = df.columns[cols[0]]
    meta_col1 = df.columns[meta_cols[0]]
    col2 = df.columns[cols[1]]
    meta_col2 = df.columns[meta_cols[1]]
    four_col_df = df[[col1, col2, meta_col1, meta_col2]].drop_duplicates()
    four_col_df[col_name + '_a'] = four_col_df.apply(lambda row: (row[col1] if row[col1] <= row[col2]
                                                                  else row[col2]), axis=1)
    four_col_df[col_name + '_a_' + meta_name] = four_col_df.apply(lambda row: (row[meta_col1] if row[col1] <= row[col2]
                                                                  else row[meta_col2]), axis=1)
    four_col_df[col_name + '_b'] = four_col_df.apply(lambda row: (row[col2] if row[col1] <= row[col2]
                                                                  else row[col1]), axis=1)
    four_col_df[col_name + '_b_' + meta_name] = four_col_df.apply(lambda row: (row[meta_col2] if row[col1] <= row[col2]
                                                                  else row[meta_col1]), axis=1)
    ordered_df = df.merge(four_col_df, how='inner', on=[col1, meta_col1, col2, meta_col2])
    return ordered_df


def aggregate_guide_lfcs(df):
    agg_lfcs = (df.groupby(['guide_a', 'guide_b', 'guide_a_gene', 'guide_b_gene', 'condition'])
                .agg({'lfc': 'mean'})
                .reset_index())
    return agg_lfcs


def build_anchor_df(df):
    """Build a dataframe where each guide is defined as an anchor and is paired with all other target guides

    Parameters
    ----------
    df: DataFrame
        LFC df

    Returns
    -------
    DataFrame
    """
    df_columns = df.columns
    forward_df = (df
                  .rename({df_columns[0]: 'anchor_guide', df_columns[1]: 'target_guide',
                           df_columns[2]: 'anchor_gene', df_columns[3]: 'target_gene'}, axis=1))
    reverse_df = (df
                  .rename({df_columns[1]: 'anchor_guide', df_columns[0]: 'target_guide',
                           df_columns[3]: 'anchor_gene', df_columns[2]: 'target_gene'}, axis=1))
    anchor_df = (pd.concat([forward_df, reverse_df])
                 .drop_duplicates()  # in case where guide1==guide2
                 .reset_index(drop=True))
    return anchor_df


def get_base_score(df, ctl_genes):
    """Get LFC of each guide when paired with controls

    Parameters
    ----------
    df: DataFrame
        Anchor guides paired with all of their target guides. Has columns "anchor_guide" and "lfc"
    ctl_genes: list
        Control genes

    Returns
    -------
    DataFrame
        Median LFC for each anchor guide
    """
    base_score = (df[df.target_gene.isin(ctl_genes)]
                  .groupby(['anchor_guide', 'condition'])
                  .agg({'lfc': 'median'})
                  .reset_index())
    return base_score


def get_no_control_guides(df, guide_base_score):
    """Get guides that are not paired with controls

    Parameters
    ----------
    df: DataFrame
        DataFrame with the column anchor_gene
    guide_base_score: DataFrame
        Guide scores when paired with controls

    Returns
    -------
    list
        Guides without control pairs
    """
    starting_guides = set(df.anchor_guide.to_list())
    base_score_guides = set(guide_base_score.anchor_guide.to_list())
    no_control_guides = list(starting_guides - base_score_guides)
    return no_control_guides


def get_removed_guides_genes(anchor_df, guides):
    """Get dataframe of removed guides and genes

    Parameters
    ----------
    anchor_df: DataFrame
    guides: list
        List of guides being removed

    Returns
    -------
    DataFrame
    """
    remove_anchor = (anchor_df.loc[anchor_df.anchor_guide.isin(guides),
                                   ['anchor_guide', 'anchor_gene']]
                     .rename({'anchor_guide': 'guide', 'anchor_gene': 'gene'}, axis=1)
                     .drop_duplicates())
    return remove_anchor


def warn_no_control_guides(anchor_df, no_control_guides):
    """Give warning for guides with control pairs

    Parameters
    ----------
    anchor_df: DataFrame
    no_control_guides: list

    Warnings
    --------
    Guides without control pairs
    """
    if len(no_control_guides) > 0:
        removed_genes = get_removed_guides_genes(anchor_df, no_control_guides)
        warnings.warn('There are ' + str(len(no_control_guides)) + ' guides without control pairs:\n' +
                      str(removed_genes), stacklevel=2)


def join_anchor_base_score(anchor_df, base_df):
    """Join anchor DataFrame with Base LFCs on anchor_guide

    Parameters
    ----------
    anchor_df: DataFrame
        DataFrame with anchor and target guides
    base_df: DataFrame
        Base LFC DataFrame

    Returns
    -------
    DataFrame
    """
    joined_df = (anchor_df.merge(base_df, how='inner', left_on=['target_guide', 'condition'],
                                 right_on=['anchor_guide', 'condition'], suffixes=['', '_target'])
                 .drop('anchor_guide_target', axis=1))
    return joined_df


def check_min_guide_pairs(df, min_pairs):
    """Check that each guide is paired with a minimum number of guides

    Parameters
    ----------
    df: DataFrame
        Anchor df with column anchor_guide
    min_pairs: int
        minimum number of guides to be paired with

    Returns
    -------
    List
        guides that are not paired fewer than min_pairs number of guides
    """
    pair_count = (df[['anchor_guide', 'target_guide']]
                  .drop_duplicates()
                  .groupby('anchor_guide')
                  .apply(lambda d: d.shape[0])
                  .reset_index(name='n'))
    guides_no_pairs = pair_count.anchor_guide[~(pair_count.n >= min_pairs)].to_list()
    return guides_no_pairs


def remove_guides(df, rm_guides):
    """Remove guides from DataFrame

    Parameters
    ----------
    df: DataFrame
        Dataframe with columns for guides
    rm_guides: list
        List of guides to remove

    Returns
    -------
    DataFrame
        Filtered to remove guides
    """
    df = df[~(df.anchor_guide.isin(rm_guides))]
    return df


def filter_anchor_base_scores(df, min_pairs):
    """Filter guides that are not in with a certain number of pairs

    Parameters
    ----------
    df: DataFrame
        DataFrame with column anchor_guide
    min_pairs: int
        Minimum number of pairs for an anchor_guide

    Returns
    -------
    DataFrame
        Filtered dataframe if there are guides without the minimum number of guide pairs
    """
    guides_no_pairs = check_min_guide_pairs(df, min_pairs)
    if len(guides_no_pairs) > 0:
        removed_genes = get_removed_guides_genes(df, guides_no_pairs)
        warnings.warn('Removed ' + str(len(guides_no_pairs)) + ' guides with fewer than ' +
                      str(min_pairs) + ' pairs:\n' + str(removed_genes), stacklevel=2)
        filtered_anchor_base_scores = remove_guides(df, guides_no_pairs)
        return filtered_anchor_base_scores
    return df


def model_fixed_slope(train_x, train_y, test_x, slope=1):
    """Predict guide phenotype using fixed slope

    From: https://stackoverflow.com/questions/33292969/linear-regression-with-specified-slope

    Parameters
    ----------
    train_x: Series
        Single gene phenotype for training genes
    train_y: Series
        Pair phenotype for training genes
    test_x: Series
        Single gene phenotype for testing genes
    slope: int
        Slope to fit model

    Returns
    -------
    Series
        Predicted phenotype of pair
    DataFrame
        Information about model
    """
    intercept = np.mean(train_y - train_x*slope)
    model_info = {'model': 'fixed_slope', 'const': intercept}
    predictions = test_x*slope + intercept
    return predictions, model_info


def model_spline(train_x, train_y, test_x, df=4):
    """Predict guide phenotype using a natural cubic spline

    Parameters
    ----------
    train_x: Series
        Single gene phenotype for training genes
    train_y: Series
        Pair phenotype for training genes
    test_x: Series
        Single gene phenotype for testing genes

    Returns
    -------
    Series
        Predicted phenotype of pair
    DataFrame
        Information about GLM model
    """
    train_x = train_x.rename('x', axis=1)
    train_x = sm.add_constant(train_x)
    train_df = train_x.copy()
    train_df['y'] = train_y
    model_fit = sm.formula.ols('y ~ cr(x, df=' + str(df) + ') + const', data=train_df).fit()
    model_info = {'model': 'spline', 'const': model_fit.params.const}
    test_x = test_x.rename('x')
    test_x = sm.add_constant(test_x)
    predictions = model_fit.predict(test_x)
    return predictions, model_info


def model_linear(train_x, train_y, test_x):
    """Predict guide phenotype using a linear model and ordinary least squares

    Parameters
    ----------
    train_x: Series
        Single gene phenotype for training genes
    train_y: Series
        Pair phenotype for training genes
    test_x: Series
        Single gene phenotype for testing genes

    Returns
    -------
    Series
        Predicted phenotype of pair
    DataFrame
        Information about OLS model
    """
    train_x = sm.add_constant(train_x)
    model_fit = sm.OLS(train_y, train_x).fit()
    model_info = {'model': 'linear', 'R2': model_fit.rsquared, 'f_pvalue': model_fit.f_pvalue,
                  'const': model_fit.params.const, 'beta': model_fit.params.values[1]}
    predictions = model_fit.predict(sm.add_constant(test_x))
    return predictions, model_info


def model_quadratic(train_x, train_y, test_x):
    """Predict guide phenotype using a linear model and ordinary least squares

    Parameters
    ----------
    train_x: Series
        Single gene phenotype for training genes
    train_y: Series
        Pair phenotype for training genes
    test_x: Series
        Single gene phenotype for testing genes

    Returns
    -------
    Series
        Predicted phenotype of pair
    DataFrame
        Information about OLS model
    """
    train_x = train_x.rename('x', axis=1)
    train_x = sm.add_constant(train_x)
    train_df = train_x.copy()
    train_df['y'] = train_y
    model_fit = sm.formula.ols('y ~ np.power(x, 2) + x + const', data=train_df).fit()
    model_info = {'model': 'quadratic', 'R2': model_fit.rsquared, 'f_pvalue': model_fit.f_pvalue,
                  'const': model_fit.params.const}
    test_x = test_x.rename('x')
    test_x = sm.add_constant(test_x)
    predictions = model_fit.predict(test_x)
    return predictions, model_info


def fit_anchor_model(df, fit_genes, model, x_col='lfc_target', y_col='lfc'):
    """Fit linear model for a single anchor guide paired with all target guides in a condition

    Parameters
    ----------
    df: DataFrame
        LFCs for a single anchor anchor guide
    fit_genes: list
        Genes used to fit the linear model
    model: str
        Name of model used to fit x and y data
    x_col: str, optional
        X column to fit model
    y_col: str, optional
        Y column to fit model

    Returns
    -------
    DataFrame:
        Guide level residuals
    DataFrame:
        R^2 for model
    """
    if fit_genes is not None:
        train_df = df.loc[df.target_gene.isin(fit_genes), :].copy()
    else:
        train_df = df
    train_x = train_df[x_col].copy()
    train_y = train_df[y_col].copy()
    test_x = df[x_col].copy()
    test_y = df[y_col].copy()
    if model == 'linear':
        predictions, model_info = model_linear(train_x, train_y, test_x)
    elif model == 'fixed slope':
        predictions, model_info = model_fixed_slope(train_x, train_y, test_x)
    elif model == 'spline':
        predictions, model_info = model_spline(train_x, train_y, test_x)
    elif model == 'quadratic':
        predictions, model_info = model_quadratic(train_x, train_y, test_x)
    else:
        raise ValueError('Model ' + model + ' not implemented')
    out_df = df.copy()
    out_df['residual'] = test_y - predictions
    out_df['residual_z'] = (out_df['residual'] - out_df['residual'].mean())/out_df['residual'].std()
    return out_df, model_info


def fit_models(df, fit_genes, model):
    """Loop through anchor guides and fit a linear model

    Parameters
    ----------
    df: DataFrame
        LFCs for all anchor guides
    fit_genes: list
        Genes used to fit the linear model
    model: str
        Name of model used to fit x and y data

    Returns
    -------
    DataFrame:
        Guide level residuals
    DataFrame:
        R^2 for each model
    """
    model_info_list = []
    residual_list = []
    for guide_condition, group_df in df.groupby(['anchor_guide', 'condition']):
        residuals, model_info = fit_anchor_model(group_df, fit_genes, model)
        residual_list.append(residuals)
        model_info['anchor_guide'] = guide_condition[0]
        model_info['anchor_gene'] = group_df['anchor_gene'].values[0]
        model_info['condition'] = guide_condition[1]
        model_info_list.append(model_info)
    model_info_df = pd.DataFrame(model_info_list)
    residual_df = (pd.concat(residual_list, axis=0)
                   .reset_index(drop=True))
    return residual_df, model_info_df


def get_guide_residuals(lfc_df, ctl_genes, fit_genes=None,
                        min_pairs=5, model='linear'):
    """Calculate guide-level residuals

    Parameters
    ----------
    lfc_df: DataFrame
        LFCs from combinatorial screen
        * Column 1 - first guide identifier
        * Column 2 - second guide identifier
        * Column 3 - first gene identifier
        * Column 4 - second gene identifier
        * Column 5+ - LFC values from different conditions
    ctl_genes: list
        Negative control genes (e.g. nonessential, intronic, or no site)
    fit_genes: list, optional
        Genes used to train each linear model. If None, uses all genes to fit. This can be helpful if we expect
        a large fraction of target_genes to be interactors
    min_pairs: int, optional
        Minimum number of pairs a guide must be in
    model: str, optional
        Name of model to fit to data

    Returns
    -------
    DataFrame
        Guide level residuals
    DataFrame
        R-squared and f-statistic p-value for each linear model
    """
    check_guide_input(lfc_df)
    melted_lfc_df = melt_df(lfc_df)
    reordered_guides = order_cols_with_meta(melted_lfc_df, [0, 1], [2, 3], 'guide', 'gene')
    dedup_guide_lfcs = aggregate_guide_lfcs(reordered_guides)
    melted_anchor_df = build_anchor_df(dedup_guide_lfcs)
    guide_base_score = get_base_score(melted_anchor_df, ctl_genes)
    no_control_guides = get_no_control_guides(melted_anchor_df, guide_base_score)
    warn_no_control_guides(melted_anchor_df, no_control_guides)
    anchor_base_scores = join_anchor_base_score(melted_anchor_df, guide_base_score)
    filtered_anchor_base_scores = filter_anchor_base_scores(anchor_base_scores, min_pairs)
    guide_residuals, model_info_df = fit_models(filtered_anchor_base_scores, fit_genes, model)
    return guide_residuals, model_info_df


def get_pop_stats(df, stat):
    """Get mean and standard deviation for a stat

    Parameters
    ----------
    df: DataFrame
        Guide level scores
    stat: str
        Column to caclulate statistics from

    Returns
    -------
    DataFrame
        Mean and standard deviation of the specified stat
    """
    pop_stats = (df.groupby('condition')
                 .agg(mean_stat=(stat, 'mean'),
                      std_stat=(stat, 'std'))
                 .reset_index())
    return pop_stats


def combine_z_scores(df, stat):
    """Combine z-score for a gene pair

    Sum z-scores and divide by the square root of the number of observations

    Parameters
    ----------
    df: DataFrame
        guide level statistic
    stat: str
        statistic to combine

    Returns
    -------
    DataFrame
        z_score for gene combination
    """
    guide1 = df.columns[0]
    guide2 = df.columns[1]
    combo_stats = (df.groupby(['condition', 'gene_a', 'gene_b'])
                   .agg(sum_stat=(stat, 'sum'))
                   .reset_index())
    guide_pair_df = (df.groupby(['condition', 'gene_a', 'gene_b'])
                     .apply(lambda d: len({frozenset(x) for x in zip(d[guide1], d[guide2])}))  # use sets to count
                     # # unique guide pairs
                     .reset_index(name='guide_pairs'))
    combo_stats = combo_stats.merge(guide_pair_df, how='inner', on=['condition', 'gene_a', 'gene_b'])
    combo_stats['pair_z_score'] = (combo_stats['sum_stat'] / np.sqrt(combo_stats['guide_pairs']))
    return combo_stats[['condition', 'gene_a', 'gene_b', 'guide_pairs', 'pair_z_score']]


def get_avg_score(df, score):
    """Get avg score for gene pairs

    Parameters
    ----------
    df: DataFrame
        Guide-level DataFrame
    score: str
        Column to average

    Returns
    -------
    DataFrame
        Average score for gene pairs
    """
    avg_score = (df.groupby(['condition', 'gene_a', 'gene_b'])
                 .agg({score: 'mean'})
                 .reset_index())
    return avg_score


def get_base_lfc_from_resid(residual_df):
    """Calculate gene level base lfcs from the guide-level residual dataframe

    Parameters
    ----------
    residual_df: DataFrame
        DataFrame of residuals

    Returns
    -------
    DataFrame
        With columns gene, condition, base_lfc
    """
    gene_base_lfcs = (residual_df[['target_gene', 'condition', 'lfc_target']]
                      .drop_duplicates()
                      .groupby(['target_gene', 'condition'])
                      .agg({'lfc_target': 'mean'})
                      .reset_index()
                      .rename({'lfc_target': 'base_lfc', 'target_gene': 'gene'}, axis=1))
    return gene_base_lfcs


def get_gene_residuals(guide_residuals, stat='residual_z'):
    """Combine residuals at the gene level

    Parameters
    ----------
    guide_residuals: DataFrame
        Guide level residuals
    stat: str, optional
        Statistic to combine at the gene level

    Returns
    -------
    DataFrame
        Gene level z_scores
    """
    ordered_df = order_cols(guide_residuals, [2, 3], 'gene')
    gene_a_anchor_z = combine_z_scores(ordered_df[ordered_df.gene_a == ordered_df.anchor_gene], stat)
    gene_b_anchor_z = combine_z_scores(ordered_df[ordered_df.gene_b == ordered_df.anchor_gene], stat)
    avg_lfc = get_avg_score(ordered_df, 'lfc')
    base_lfcs = get_base_lfc_from_resid(guide_residuals)
    gene_results = (avg_lfc.merge(base_lfcs, how='inner', left_on=['condition', 'gene_a'],
                                  right_on=['condition', 'gene'])
                    .drop('gene', axis=1)
                    .merge(base_lfcs, how='inner', left_on=['condition', 'gene_b'],
                           right_on=['condition', 'gene'], suffixes=['_a', '_b'])
                    .drop('gene', axis=1)
                    .merge(gene_a_anchor_z, how='inner',
                           on=['condition', 'gene_a', 'gene_b'])
                    .merge(gene_b_anchor_z, how='inner',
                           on=['condition', 'gene_a', 'gene_b', 'guide_pairs'], suffixes=['_gene_a_anchor',
                                                                                          '_gene_b_anchor']))
    gene_results['pair_z_score'] = (gene_results[['pair_z_score_gene_a_anchor', 'pair_z_score_gene_b_anchor']]
                                    .sum(axis=1) / np.sqrt(2))
    return gene_results


def calc_dlfc(df, base_lfcs):
    """Add together base lfcs to generate an expectation for each guide pair

    Parameters
    ----------
    df: DataFrame
        Combo level LFCs
    base_lfcs: DataFrame
        Base LFCs - single guide phenotype

    Returns
    -------
    DataFrame
        delta LFCs for each guide pair
    """
    guide1 = df.columns[0]
    guide2 = df.columns[1]
    dlfc = (df.merge(base_lfcs, how='inner', left_on=[guide1, 'condition'],
                     right_on=['anchor_guide', 'condition'], suffixes=['', '_' + guide1 + '_base'])
            .drop('anchor_guide', axis=1)
            .merge(base_lfcs, how='inner', left_on=[guide2, 'condition'],
                   right_on=['anchor_guide', 'condition'], suffixes=['', '_' + guide2 + '_base'])
            .drop('anchor_guide', axis=1))
    dlfc['sum_lfc'] = dlfc['lfc_' + guide1 + '_base'] + dlfc['lfc_' + guide2 + '_base']
    dlfc['dlfc'] = dlfc['lfc'] - dlfc['sum_lfc']
    dlfc['dlfc_z'] = (dlfc.groupby('condition')
                      .dlfc
                      .transform(lambda x: (x - x.mean())/x.std()))
    return dlfc


def get_guide_dlfcs(lfc_df, ctl_genes):
    """Calculate delta-LFC's at the guide level

    Model the LFC of each combination as the sum of each guide when paired with controls. The difference from this
    expectation is the delta log2-fold change

    Parameters
    ----------
    lfc_df: DataFrame
        LFCs from combinatorial screen

        - Column 1: first guide identifier
        - Column 2: second guide identifier
        - Column 3: first gene identifier
        - Column 4: second gene identifier
        - Column 5+: LFC values from different conditions
    ctl_genes: list
        Negative control genes (e.g. nonessential, intronic, or no site)

    Returns
    -------
    DataFrame:
        delta LFCs for guide pairs
    """
    check_guide_input(lfc_df)
    #  get base lfcs using anchor framework
    melted_lfc_df = melt_df(lfc_df)
    reordered_guides = order_cols_with_meta(melted_lfc_df, [0, 1], [2, 3], 'guide', 'gene')
    dedup_guide_lfcs = aggregate_guide_lfcs(reordered_guides)
    melted_anchor_df = build_anchor_df(dedup_guide_lfcs)
    guide_base_score = get_base_score(melted_anchor_df, ctl_genes)
    no_control_guides = get_no_control_guides(melted_anchor_df, guide_base_score)
    #  calculate delta-log-fold changes
    warn_no_control_guides(melted_anchor_df, no_control_guides)
    melted_df = melt_df(lfc_df)
    guide_dlfc = calc_dlfc(melted_df, guide_base_score)
    return guide_dlfc


def get_base_lfc_from_dlfc(dlfc_df):
    """Calculate gene level base lfcs from the guide-level dLFC dataframe

    Parameters
    ----------
    dlfc_df: DataFrame
        DataFrame of delta log-fold changes

    Returns
    -------
    DataFrame
        With columns gene, condition, base_lfc
    """
    gene1 = dlfc_df.columns[2]
    gene2 = dlfc_df.columns[3]
    guide1_base = 'lfc_' + dlfc_df.columns[0] + '_base'
    guide2_base = 'lfc_' + dlfc_df.columns[1] + '_base'
    base_lfcs = (pd.concat([dlfc_df.rename({gene1: 'gene', guide1_base: 'base_lfc'}, axis=1),
                            dlfc_df.rename({gene2: 'gene', guide2_base: 'base_lfc'}, axis=1)])
                 .groupby(['condition', 'gene'])
                 .agg({'base_lfc': 'mean'})
                 .reset_index())
    return base_lfcs


def get_gene_dlfcs(guide_dlfcs, stat='dlfc_z'):
    """Combine dLFCs at the gene level

    Parameters
    ----------
    guide_dlfcs: DataFrame
        Guide level dLFCs
    stat: str, optional
        Statistic to combine at the gene level

    Returns
    -------
    DataFrame
        Gene level z_scores
    """
    ordered_df = order_cols(guide_dlfcs, [2, 3], 'gene')
    combined_z = combine_z_scores(ordered_df, stat)
    avg_lfc = get_avg_score(ordered_df, 'lfc')
    base_lfcs = get_base_lfc_from_dlfc(guide_dlfcs)
    gene_results = (avg_lfc.merge(base_lfcs, how='inner', left_on=['condition', 'gene_a'],
                                  right_on=['condition', 'gene'])
                    .drop('gene', axis=1)
                    .merge(base_lfcs, how='inner', left_on=['condition', 'gene_b'],
                           right_on=['condition', 'gene'], suffixes=['_a', '_b'])
                    .drop('gene', axis=1)
                    .merge(combined_z, how='inner', on=['condition', 'gene_a', 'gene_b']))
    return gene_results
