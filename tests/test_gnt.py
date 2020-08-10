#!/usr/bin/env python

"""Tests for `gnt` package."""

import pytest

from click.testing import CliRunner

from gnt import cli
from gnt import score
import gnt
import pandas as pd
import warnings
import numpy as np


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli.main, [
            "https://raw.githubusercontent.com/PeterDeWeirdt/bigpapi/master/data/processed/bigpapi_lfcs.csv",
            "test", "--control", "CD81", "--control", "HPRT intron"])
        assert result.exit_code == 0
        output_gene_file = pd.read_csv('test_gnt_residual_gene_scores.csv')
        assert ((output_gene_file.sort_values('pair_z_score')
                 .head(1)
                 [['gene_a', 'gene_b']]
                 .values) == [['MAPK1', 'MAPK3']]).all()
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0


@pytest.fixture
def bigpapi_lfcs():
    df = pd.read_csv('https://raw.githubusercontent.com/PeterDeWeirdt/bigpapi/master/data/processed/bigpapi_lfcs.csv')
    return df


def test_build_anchor_df(bigpapi_lfcs):
    anchor_df = score.build_anchor_df(bigpapi_lfcs)
    assert anchor_df.shape[0] == bigpapi_lfcs.shape[0]*2
    assert anchor_df.anchor_guide.value_counts().median() == (25*3+3+10+5+3)
    toy_df = pd.DataFrame({'guide1': ['a', 'a', 'b'], 'guide2': ['a', 'a', 'a'], 'gene1': ['A', 'A', 'B'],
                           'gene2': ['A', 'A', 'A']})
    toy_anchor_df = score.build_anchor_df(toy_df)
    expected_output = pd.DataFrame({'anchor_guide': ['a', 'b', 'a'], 'target_guide': ['a', 'a', 'b'],
                                    'anchor_gene': ['A', 'B', 'A'], 'target_gene': ['A', 'A', 'B']})
    pd.testing.assert_frame_equal(toy_anchor_df, expected_output)


def test_guide_base_score(bigpapi_lfcs):
    anchor_df = score.build_anchor_df(bigpapi_lfcs)
    melted_anchor_df = score.melt_df(anchor_df)
    guide_base_score = score.get_base_score(melted_anchor_df, ['HPRT intron', 'CD81'])
    assert guide_base_score.shape[0] == 96*2*6
    assert guide_base_score.sort_values('lfc').anchor_guide.values[0] == 'TGGTGTGCAAGGCGGGCATCA'  # EEF2 guide


def test_fit_anchor_model(bigpapi_lfcs):
    anchor_df = score.build_anchor_df(bigpapi_lfcs)
    melted_anchor_df = score.melt_df(anchor_df)
    guide_base_score = score.get_base_score(melted_anchor_df, ['HPRT intron', 'CD81'])
    anchor_base_scores = score.join_anchor_base_score(melted_anchor_df, guide_base_score)
    train_df = anchor_base_scores.loc[(anchor_base_scores.anchor_guide == 'GCTGTATCCTTTCTGGGAAAG') &
                                      (anchor_base_scores.condition == 'Day 21_Meljuso'), :]  # BCL2L1 guide
    residuals, model_info = score.fit_anchor_model(train_df, None, 'linear')
    assert model_info['R2'] > 0.5
    gene_residuals = (residuals.groupby('target_gene')
                      .agg({'residual_z': 'mean'})
                      .sort_values('residual_z')
                      .reset_index())
    assert gene_residuals.loc[0, 'target_gene'] == 'MCL1'
    assert gene_residuals['target_gene'].iloc[-1] == 'BCL2L1'
    _, ctl_model_info = score.fit_anchor_model(train_df, ['CD81', 'HPRT intron'], 'linear')
    assert model_info['f_pvalue'] < ctl_model_info['f_pvalue']


def test_get_guide_residuals(bigpapi_lfcs):
    guide_residuals, model_info_df = gnt.get_guide_residuals(bigpapi_lfcs, ['CD81', 'HPRT intron'])
    assert ((guide_residuals.groupby(['anchor_gene', 'target_gene', 'condition'])
             .agg({'residual_z': 'mean'})
             .sort_values('residual_z')
             .reset_index()
             .head(1)
             [['anchor_gene', 'target_gene']]
             .values) == [['MAPK3', 'MAPK1']]).all()


def test_order_guides():
    guides = pd.DataFrame({'guide1': ['a', 'a', 'a', 'b'], 'guide2': ['a', 'a', 'b', 'a'],
                          'anchor_gene': [1, 2, 3, 4], 'target_gene': [1, 2, 3, 4]})
    ordered_genes = score.order_cols(guides, [0, 1], 'guide')
    expected_order = guides.copy()
    expected_order['guide_a'] = ['a', 'a', 'a', 'a']
    expected_order['guide_b'] = ['a', 'a', 'b', 'b']
    pd.testing.assert_frame_equal(ordered_genes, expected_order)


def test_guide_deduplication(bigpapi_lfcs):
    rev_bigpapi_lfcs = (bigpapi_lfcs.rename({'U6 Sequence': 'H1 Sequence',
                                             'H1 Sequence': 'U6 Sequence',
                                             'U6 gene': 'H1 gene',
                                             'H1 gene': 'U6 gene'}, axis=1))
    lfc_df = pd.concat([bigpapi_lfcs, rev_bigpapi_lfcs], axis=0)
    melted_lfc_df = gnt.score.melt_df(lfc_df)
    reordered_guides = gnt.score.order_cols_with_meta(melted_lfc_df, [0, 1], [2, 3], 'guide', 'gene')
    dedup_guide_lfcs = gnt.score.aggregate_guide_lfcs(reordered_guides)
    assert dedup_guide_lfcs.shape[0] == bigpapi_lfcs.shape[0]*6
    possible_guide_gene_pairs = set((bigpapi_lfcs['U6 Sequence'] + bigpapi_lfcs['U6 gene']).to_list() +
                                    (bigpapi_lfcs['H1 Sequence'] + bigpapi_lfcs['H1 gene']).to_list())
    observed_guide_gene_pairs = set((dedup_guide_lfcs['guide_a'] +
                                     dedup_guide_lfcs['guide_a_gene']).to_list() +
                                    (dedup_guide_lfcs['guide_b'] +
                                     dedup_guide_lfcs['guide_b_gene']).to_list())
    assert possible_guide_gene_pairs == observed_guide_gene_pairs


def test_order_genes():
    genes = pd.DataFrame({'guide1': [1, 2, 3], 'guide2': [1, 2, 3],
                          'anchor_gene': ['A', 'B', 'B'], 'target_gene': ['B', 'B', 'A']})
    ordered_genes = score.order_cols(genes, [2, 3], 'gene')
    expected_order = genes.copy()
    expected_order['gene_a'] = ['A', 'B', 'A']
    expected_order['gene_b'] = ['B', 'B', 'B']
    pd.testing.assert_frame_equal(ordered_genes, expected_order)


def test_get_gene_residuals(bigpapi_lfcs):
    guide_residuals, model_info_df = gnt.get_guide_residuals(bigpapi_lfcs, ['CD81', 'HPRT intron'])
    gene_results = gnt.get_gene_residuals(guide_residuals, 'residual_z')
    assert ((gene_results.sort_values('pair_z_score')
             .head(1)
             [['gene_a', 'gene_b']]
             .values) == [['MAPK1', 'MAPK3']]).all()
    gene_results = gnt.get_gene_residuals(guide_residuals, 'residual_z')
    assert ((gene_results.sort_values('pair_z_score')
             .head(1)
             [['gene_a', 'gene_b']]
             .values) == [['MAPK1', 'MAPK3']]).all()


def test_get_guide_dlfc(bigpapi_lfcs):
    guide_dlfcs = gnt.get_guide_dlfcs(bigpapi_lfcs, ['HPRT intron', 'CD81'])
    assert ((guide_dlfcs.groupby(['U6 gene', 'H1 gene', 'condition'])
             .agg({'dlfc_z': 'mean'})
             .sort_values('dlfc_z')
             .reset_index()
             .head(1)
             [['U6 gene', 'H1 gene']]
             .values) == [['MCL1', 'BCL2L1']]).all()


def test_get_gene_dlfc(bigpapi_lfcs):
    guide_dlfcs = gnt.get_guide_dlfcs(bigpapi_lfcs, ['HPRT intron', 'CD81'])
    gene_dlfc = gnt.get_gene_dlfcs(guide_dlfcs, 'dlfc')
    assert ((gene_dlfc[gene_dlfc.guide_pairs < 20]
             .sort_values('pair_z_score')
             .head(1)
             [['gene_a', 'gene_b']]
             .values) == [['MAPK1', 'MAPK3']]).all()


def test_get_base_lfc_from_resid(bigpapi_lfcs):
    guide_residuals, _ = gnt.get_guide_residuals(bigpapi_lfcs, ['HPRT intron', 'CD81'])
    gene_base_lfcs = gnt.score.get_base_lfc_from_resid(guide_residuals)
    assert ((gene_base_lfcs
             .sort_values('base_lfc')
             ['gene']
             .values[0]) == 'EEF2')


def test_warn_controls(bigpapi_lfcs):
    ctl_genes = ['CD81', 'HPRT intron']
    bcl2l1_no_control = bigpapi_lfcs[~(((bigpapi_lfcs.iloc[:, 2] == 'BCL2L1') &
                                        bigpapi_lfcs.iloc[:, 3].isin(ctl_genes)) |
                                       ((bigpapi_lfcs.iloc[:, 3] == 'BCL2L1') &
                                        bigpapi_lfcs.iloc[:, 2].isin(ctl_genes)))]
    anchor_df = gnt.score.build_anchor_df(bcl2l1_no_control)
    melted_anchor_df = gnt.score.melt_df(anchor_df, None)
    guide_base_score = gnt.score.get_base_score(melted_anchor_df, ctl_genes)
    no_control_guides = gnt.score.get_no_control_guides(melted_anchor_df, guide_base_score)
    assert len(no_control_guides) == 6  # 3 Sa and 3 Sp
    with warnings.catch_warnings(record=True) as w:
        gnt.score.warn_no_control_guides(anchor_df, no_control_guides)
        assert len(w) == 1
        assert '6' in str(w[0].message)


def test_filter_anchor_base_scores(bigpapi_lfcs):
    ctl_genes = ['CD81', 'HPRT intron']
    bcl2l1_no_pairs = bigpapi_lfcs[~(((bigpapi_lfcs.iloc[:, 2] == 'BCL2L1') &
                                      ~(bigpapi_lfcs.iloc[:, 1] == 'AAAAAAAGAGTCGAATGTTTT')) |
                                     ((bigpapi_lfcs.iloc[:, 3] == 'BCL2L1') &
                                      ~(bigpapi_lfcs.iloc[:, 0] == 'AAAGTGGAACTCAGGACATG')))]
    anchor_df = gnt.score.build_anchor_df(bcl2l1_no_pairs)
    melted_anchor_df = gnt.score.melt_df(anchor_df, None)
    guide_base_score = gnt.score.get_base_score(melted_anchor_df, ctl_genes)
    anchor_base_scores = gnt.score.join_anchor_base_score(melted_anchor_df, guide_base_score)
    guides_no_pairs = gnt.score.check_min_guide_pairs(anchor_base_scores, 5)
    assert len(guides_no_pairs) == 6
    with warnings.catch_warnings(record=True) as w:
        filtered_anchor_base_scores = gnt.score.filter_anchor_base_scores(anchor_base_scores, 5)
        assert len(w) == 1
        assert '6' in str(w[0].message)
        assert filtered_anchor_base_scores.shape[0] < anchor_base_scores.shape[0]
        assert filtered_anchor_base_scores.shape[0] > 0


def test_model_fixed_slope():
    train_x = pd.Series([1, 2, 3])
    train_y = pd.Series([2, 3, 4])
    predictions, model_info = score.model_fixed_slope(train_x, train_y, train_x, slope=1)
    assert (predictions == train_y).all()
    assert model_info['const'] == 1


def test_model_quadratic_ols():
    train_x = pd.Series([-2, -1, 0, 1, 2])
    train_y = pd.Series([4, 1, 0, 1, 4])
    predictions, model_info = score.model_quadratic(train_x, train_y, train_x)
    assert np.allclose(predictions, train_y)


def test_model_spline_glm():
    train_x = pd.Series([-2, -1, 0, 1, 2])
    train_y = pd.Series([2, 2, 1, 0.5, 1])
    spline_predictions, _ = score.model_spline(train_x, train_y, train_x)
    spline_residual = (spline_predictions - train_y).abs().mean()
    quad_predictions, _ = score.model_quadratic(train_x, train_y, train_x)
    quad_residual = (quad_predictions - train_y).abs().mean()
    linear_predictions, _ = score.model_linear(train_x, train_y, train_x)
    linear_residual = (linear_predictions - train_y).abs().mean()
    assert spline_residual < linear_residual
    assert spline_residual < quad_residual
