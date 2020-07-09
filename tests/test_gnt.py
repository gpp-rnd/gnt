#!/usr/bin/env python

"""Tests for `gnt` package."""

import pytest

from click.testing import CliRunner

from gnt import cli
from gnt import score
import gnt
import pandas as pd


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli.main, ["https://raw.githubusercontent.com/PeterDeWeirdt/bigpapi/master/data/processed/bigpapi_lfcs.csv",
                                          "test", "--control", "CD81", "--control", "HPRT intron"])
        assert result.exit_code == 0
        output_gene_file = pd.read_csv('test_gnt_residual_gene_scores.csv')
        assert ((output_gene_file.sort_values('z_score_residual_z')
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
    residuals, model_info = score.fit_anchor_model(train_df, None, False)
    assert model_info['R2'] > 0.5
    gene_residuals = (residuals.groupby('target_gene')
                      .agg({'residual_z': 'mean'})
                      .sort_values('residual_z')
                      .reset_index())
    assert gene_residuals.loc[0, 'target_gene'] == 'MCL1'
    assert gene_residuals['target_gene'].iloc[-1] == 'BCL2L1'
    # test scale
    residuals, model_info = score.fit_anchor_model(train_df, None, True)
    residuals['scaled_pct_rank'] = residuals.scaled_residual_z.abs().rank(pct=True)
    residuals['unscaled_pct_rank'] = residuals.residual_z.abs().rank(pct=True)
    eef2_pct_rank = (residuals[residuals.target_gene == 'EEF2']
                     .agg({'scaled_pct_rank': 'mean',
                           'unscaled_pct_rank': 'mean'}))
    assert eef2_pct_rank['scaled_pct_rank'] < eef2_pct_rank['unscaled_pct_rank']
    _, ctl_model_info = score.fit_anchor_model(train_df, ['CD81', 'HPRT intron'], False)
    assert model_info['f_pvalue'] < ctl_model_info['f_pvalue']


def test_get_residual(bigpapi_lfcs):
    guide_residuals, model_info_df = gnt.get_guide_residuals(bigpapi_lfcs, ['CD81', 'HPRT intron'])
    assert ((guide_residuals.groupby(['anchor_gene', 'target_gene', 'condition'])
             .agg({'residual_z': 'mean'})
             .sort_values('residual_z')
             .reset_index()
             .head(1)
             [['anchor_gene', 'target_gene']]
             .values) == [['MAPK3', 'MAPK1']]).all()


def test_order_genes():
    genes = pd.DataFrame({'guide1': [1, 2, 3], 'guide2': [1, 2, 3],
                          'anchor_gene': ['A', 'B', 'B'], 'target_gene': ['B', 'B', 'A']})
    ordered_genes = score.order_genes(genes)
    expected_order = genes.copy()
    expected_order['gene_a'] = ['A', 'B', 'A']
    expected_order['gene_b'] = ['B', 'B', 'B']
    pd.testing.assert_frame_equal(ordered_genes, expected_order)


def test_get_gene_residuals(bigpapi_lfcs):
    guide_residuals, model_info_df = gnt.get_guide_residuals(bigpapi_lfcs, ['CD81', 'HPRT intron'], scale=True)
    gene_results = gnt.get_gene_residuals(guide_residuals, 'residual_z')
    assert ((gene_results.sort_values('z_score_residual_z')
             .head(1)
             [['gene_a', 'gene_b']]
             .values) == [['MAPK1', 'MAPK3']]).all()
    scaled_gene_results = gnt.get_gene_residuals(guide_residuals, 'scaled_residual_z')
    assert ((scaled_gene_results.sort_values('z_score_scaled_residual_z')
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
    assert ((gene_dlfc.sort_values('z_score_dlfc')
             .head(1)
             [['gene_a', 'gene_b']]
             .values) == [['BCL2L1', 'MCL1']]).all()
