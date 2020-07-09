"""score module. Functions to score genetic interactions"""
import pandas as pd
import statsmodels.api as sm
import numpy as np


def check_input(df):
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
                 .reset_index(drop=True))
    return anchor_df


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


def fit_anchor_model(df, fit_genes, scale, scale_alpha=0.05, x_col='lfc_target', y_col='lfc'):
    """Fit linear model for a single anchor guide paired with all target guides in a condition

    Parameters
    ----------
    df: DataFrame
        LFCs for a single anchor anchor guide
    fit_genes: list
        Genes used to fit the linear model
    scale: bool
        Whether to scale residuals by the confidence interval of the best fit line
    scale_alpha: float, optional
        Specifies width of confidence interval
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
    train_x = sm.add_constant(train_df[x_col])
    model_fit = sm.OLS(train_df[y_col], train_x).fit()
    model_info = {'R2': model_fit.rsquared, 'f_pvalue': model_fit.f_pvalue}
    test_df = df.copy()
    predictions = model_fit.predict(sm.add_constant(test_df[x_col]))
    test_df['residual'] = test_df[y_col] - predictions
    test_df['residual_z'] = (test_df['residual'] - test_df['residual'].mean())/test_df['residual'].std()
    if scale:
        prediction_results = model_fit.get_prediction(sm.add_constant(test_df[x_col]))
        summary_frame = prediction_results.summary_frame(alpha=scale_alpha)  # summary frame has confidence interval for predictions
        test_df[['mean_ci_lower', 'mean_ci_upper']] = summary_frame[['mean_ci_lower', 'mean_ci_upper']]
        test_df['ci'] = test_df['mean_ci_upper'] - test_df['mean_ci_lower']
        test_df['scaled_residual'] = test_df['residual']/test_df['ci']
        test_df['scaled_residual_z'] = (test_df['scaled_residual'] -
                                        test_df['scaled_residual'].mean()) / test_df['scaled_residual'].std()
    return test_df, model_info


def fit_models(df, fit_genes, scale_resids):
    """Loop through anchor guides and fit a linear model

    Parameters
    ----------
    df: DataFrame
        LFCs for all anchor guides
    fit_genes: list
        Genes used to fit the linear model
    scale_resids: bool
        Whether to scale residuals by the confidence interval of the best fit line

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
        residuals, model_info = fit_anchor_model(group_df, fit_genes, scale_resids)
        residual_list.append(residuals)
        model_info['anchor_guide'] = guide_condition[0]
        model_info['condition'] = guide_condition[1]
        model_info_list.append(model_info)
    model_info_df = pd.DataFrame(model_info_list)
    residual_df = (pd.concat(residual_list, axis=0)
                   .reset_index(drop=True))
    return residual_df, model_info_df


def get_guide_residuals(lfc_df, ctl_genes, fit_genes=None, scale=False):
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
    scale: bool, optional
        Whether to scale residuals by the confidence interval at a fit point (experimental). The output will then
        include a column "sclaed_residual" and "scaled_residual_z"

    Returns
    -------
    DataFrame
        Guide level residuals
    DataFrame
        R-squared and f-statistic p-value for each linear model
    """
    check_input(lfc_df)
    anchor_df = build_anchor_df(lfc_df)
    melted_anchor_df = melt_df(anchor_df, fit_genes)
    guide_base_score = get_base_score(melted_anchor_df, ctl_genes)
    anchor_base_scores = join_anchor_base_score(melted_anchor_df, guide_base_score)
    guide_residuals, model_info_df = fit_models(anchor_base_scores, fit_genes, scale)
    return guide_residuals, model_info_df


def order_genes(df):
    """Reorder anchor and target genes to be in alphabetical order

    Parameters
    ----------
    df: DataFrame
        3rd and 4th columns represent gene 1 and gene 2

    Returns
    -------
    DataFrame
        with columns gene_a and gene_b
    """
    gene1 = df.columns[2]
    gene2 = df.columns[3]
    anchor_target_df = df[[gene1, gene2]].drop_duplicates()
    anchor_target_df['gene_a'] = anchor_target_df.apply(lambda row: (row[gene1] if
                                                                     row[gene1] <= row[gene2]
                                                                     else row[gene2]),
                                                        axis=1)
    anchor_target_df['gene_b'] = anchor_target_df.apply(lambda row: (row[gene2] if
                                                                     row[gene1] <= row[gene2]
                                                                     else row[gene1]),
                                                        axis=1)
    ordered_df = df.merge(anchor_target_df, how='inner', on=[gene1, gene2])
    return ordered_df


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


def combine_statistic(df, pop_stats, stat):
    """Combine a statistic for a gene pair

    .. math::
        (x - \mu)/(\sigma / \sqrt{n})

    Where :math:`x, \mu, \sigma` are the sample mean, population mean, and population standard deviation of
    residuals, and :math:`n` is the number of guide pairs.

    Parameters
    ----------
    df: DataFrame
        guide level statistic
    pop_stats: DataFrame
        population stats
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
                   .agg(mean_stat=(stat, 'mean'))
                   .reset_index()
                   .merge(pop_stats, how='inner', on='condition',
                          suffixes=['', '_pop']))
    guide_pair_df = (df.groupby(['condition', 'gene_a', 'gene_b'])
                     .apply(lambda d: len({frozenset(x) for x in zip(d[guide1], d[guide2])}))  # use sets to count
                     # # unique guide pairs
                     .reset_index(name='guide_pairs'))
    combo_stats = combo_stats.merge(guide_pair_df, how='inner', on=['condition', 'gene_a', 'gene_b'])
    combo_stats['z_score_' + stat] = ((combo_stats['mean_stat'] - combo_stats['mean_stat_pop']) /
                                      (combo_stats['std_stat'] / np.sqrt(combo_stats['guide_pairs'])))
    return combo_stats[['condition', 'gene_a', 'gene_b', 'guide_pairs', 'z_score_' + stat]]


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


def get_gene_residuals(guide_residuals, stat='residual'):
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
    ordered_df = order_genes(guide_residuals)
    pop_stats = get_pop_stats(ordered_df, stat)
    gene_a_anchor_z = combine_statistic(ordered_df[ordered_df.gene_a == ordered_df.anchor_gene], pop_stats, stat)
    gene_b_anchor_z = combine_statistic(ordered_df[ordered_df.gene_b == ordered_df.anchor_gene], pop_stats, stat)
    combined_z = combine_statistic(ordered_df, pop_stats, stat)
    avg_lfc = get_avg_score(ordered_df, 'lfc')
    gene_results = (avg_lfc.merge(combined_z, how='inner', on=['condition', 'gene_a', 'gene_b'])
                    .merge(gene_a_anchor_z, how='inner',
                           on=['condition', 'gene_a', 'gene_b', 'guide_pairs'], suffixes=['', '_gene_a_anchor'])
                    .merge(gene_b_anchor_z, how='inner',
                           on=['condition', 'gene_a', 'gene_b', 'guide_pairs'], suffixes=['', '_gene_b_anchor']))
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
    check_input(lfc_df)
    #  get base lfcs using anchor framework
    anchor_df = build_anchor_df(lfc_df)
    melted_anchor_df = melt_df(anchor_df)
    base_lfcs = get_base_score(melted_anchor_df, ctl_genes)
    #  calculate delta-log-fold changes without generating an anchor df
    melted_df = melt_df(lfc_df)
    guide_dlfc = calc_dlfc(melted_df, base_lfcs)
    return guide_dlfc


def get_gene_dlfcs(guide_dlfcs, stat='dlfc'):
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
    ordered_df = order_genes(guide_dlfcs)
    pop_stats = get_pop_stats(ordered_df, stat)
    combined_z = combine_statistic(ordered_df, pop_stats, stat)
    avg_lfc = get_avg_score(ordered_df, 'lfc')
    gene_results = (avg_lfc.merge(combined_z, how='inner', on=['condition', 'gene_a', 'gene_b']))
    return gene_results
