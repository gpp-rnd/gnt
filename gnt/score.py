"""score module. Functions to score genetic interactions"""
import pandas as pd
import statsmodels.api as sm
import numpy as np


def check_input(df):
    """Check whether input dataframe has expected format

    * Column 1 - first guide identifier
    * Column 2 - second guide identifier
    * Column 3 - first gene identifier
    * Column 4 - second gene identifier
    * Column 5+ - LFC values from different conditions

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
    """Melt a DataFrame"""
    if id_cols is None:
        id_cols = df.columns[0:4] # anchor_guide, target_guide, anchor_gene, target_gene
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
    """Join anchor DataFrame with Base LFCs on anchor_guide"""
    joined_df = (anchor_df.merge(base_df, how='inner', left_on=['target_guide', 'condition'],
                                 right_on=['anchor_guide', 'condition'], suffixes=['', '_target'])
                 .drop('anchor_guide_target', axis=1))
    return joined_df


def fit_anchor_model(df, fit_genes, x_col='lfc_target', y_col='lfc'):
    """Fit linear model for a single anchor guide paired with all target guides in a condition"""
    if fit_genes is not None:
        train_df = df.loc[df.target_gene.isin(fit_genes), :].copy()
    else:
        train_df = df
    train_x = sm.add_constant(train_df[x_col])
    model_fit = sm.OLS(train_df[y_col], train_x).fit()
    model_info = {'R2': model_fit.rsquared, 'f_pvalue': model_fit.f_pvalue}
    if fit_genes is not None:
        test_df = df.copy()
        predictions = model_fit.predict(sm.add_constant(test_df[x_col]))
        test_df['residual'] = test_df[x_col] - predictions
    else:
        test_df = train_df.copy()
        test_df['residual'] = model_fit.resid
    test_df['residual_z'] = (test_df['residual'] - test_df['residual'].mean())/test_df['residual'].std()
    return model_info, test_df


def fit_models(df, fit_genes):
    """Fit linear model for each anchor guide in each condition"""
    model_info_list = []
    residual_list = []
    for guide_condition, group_df in df.groupby(['anchor_guide', 'condition']):
        model_info, residuals = fit_anchor_model(group_df, fit_genes)
        residual_list.append(residuals)
        model_info['anchor_guide'] = guide_condition[0]
        model_info['condition'] = guide_condition[1]
        model_info_list.append(model_info)
    model_info_df = pd.DataFrame(model_info_list)
    residual_df = (pd.concat(residual_list, axis=0)
                   .reset_index(drop=True))
    return model_info_df, residual_df


def get_residuals(df, ctl_genes, fit_genes=None):
    """Calculate guide-level residuals

    Parameters
    ----------
    df: DataFrame
        LFCs from combinatorial screen
    ctl_genes: list
        Negative control genes (e.g. nonessential, intronic, or no site)
    fit_genes: list
        Genes used to train each linear model. If None, uses all genes to fit. This can be helpful if we expect
        a large fraction of target_genes to be interactors

    Returns
    -------
    DataFrame
        R-squared and f-statistic p-value for each linear model
    DataFrame
        Guide level residuals

    """
    check_input(df)
    anchor_df = build_anchor_df(df)
    melted_anchor_df = melt_df(anchor_df, fit_genes)
    guide_base_score = get_base_score(melted_anchor_df, ctl_genes)
    anchor_base_scores = join_anchor_base_score(melted_anchor_df, guide_base_score)
    model_info_df, guide_residuals = fit_models(anchor_base_scores, fit_genes)
    return model_info_df, guide_residuals


def order_genes(df):
    """reorder anchor and target genes to be in alphabetical order"""
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


def get_pop_stats(df):
    """Get mean and standard deviation for z-scored residuals"""
    pop_stats = (df.groupby('condition')
                 .agg(mean_residual=('residual_z', 'mean'),
                      std_residual=('residual_z', 'std'))
                 .reset_index())
    return pop_stats


def combine_residuals(df, pop_stats):
    """Combine residuals for a gene pair

    .. math::
        (\bar x - \mu)/(\sigma / \sqrt{n})

    Where :math:`\bar x, \mu, \sigma` are the sample mean, population mean, and population standard deviation of
    residuals, and :math:`n` is the number of guide pairs.

    Parameters
    ----------
    df: DataFrame
        guide level residuals
    pop_stats: DataFrame
        population stats

    Returns
    -------
    DataFrame
        z_score for gene combination
    """
    combo_stats = (df.groupby(['condition', 'gene_a', 'gene_b'])
                   .agg(mean_residual=('residual_z', 'mean'),
                        count=('residual_z', 'count'))
                   .reset_index()
                   .merge(pop_stats, how='inner', on='condition',
                          suffixes=['', '_pop']))
    combo_stats['z_score'] = ((combo_stats['mean_residual'] - combo_stats['mean_residual_pop']) /
                              (combo_stats['std_residual'] / np.sqrt(combo_stats['count'])))
    return combo_stats[['condition', 'gene_a', 'gene_b', 'z_score']]


def get_avg_score(df, score):
    """get avg lfc or any other score"""
    avg_score = (df.groupby(['condition', 'gene_a', 'gene_b'])
                 .agg({score: 'mean'})
                 .reset_index())
    return avg_score


def get_gene_residuals(df):
    """Combine residuals at the gene level

    Parameters
    ----------
    df: DataFrame
        Guide level residuals

    Returns
    -------
    DataFrame
        Gene level z_scores

    """
    ordered_df = order_genes(df)
    pop_stats = get_pop_stats(ordered_df)
    gene_a_anchor_z = combine_residuals(ordered_df[ordered_df.gene_a == ordered_df.anchor_gene], pop_stats)
    gene_b_anchor_z = combine_residuals(ordered_df[ordered_df.gene_b == ordered_df.anchor_gene], pop_stats)
    combined_z = combine_residuals(ordered_df, pop_stats)
    avg_score = get_avg_score(ordered_df, 'lfc')
    gene_results = (avg_score.merge(combined_z, how='inner', on=['condition', 'gene_a', 'gene_b'])
                    .merge(gene_a_anchor_z, how='inner',
                           on=['condition', 'gene_a', 'gene_b'], suffixes=['', '_gene_a_anchor'])
                    .merge(gene_b_anchor_z, how='inner',
                           on=['condition', 'gene_a', 'gene_b'], suffixes=['', '_gene_b_anchor']))
    return gene_results


def calc_dlfc(df, base_lfcs):
    """Add together base lfcs to generate an expectation for each guide pair"""
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
    return dlfc


def get_dlfc(df, ctl_genes):
    """Calculate delta-LFC's

    Model the LFC of each combination as the sum of each guide when paired with controls. The difference from this
    expectation is the delta log2-fold change

    Parameters
    ----------
    df: DataFrame
        LFCs from combinatorial screen

    Returns
    -------
    DataFrame:
        delta LFCs for gene pairs
    DataFrame:
        delta LFCs for guide pairs
    """
    check_input(df)
    anchor_df = build_anchor_df(df)
    melted_anchor_df = melt_df(anchor_df)
    base_lfcs = get_base_score(melted_anchor_df, ctl_genes)
    melted_df = melt_df(df)
    guide_dlfc = calc_dlfc(melted_df, base_lfcs)
    ordered_dflc = order_genes(guide_dlfc)
    avg_dlfc = get_avg_score(ordered_dflc, 'dlfc')
    return avg_dlfc, guide_dlfc










