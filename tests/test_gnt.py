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
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'gnt.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


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
                                      (anchor_base_scores.condition == 'Day 21_Meljuso'), :] # BCL2L1 guide
    model_info, residuals = score.fit_anchor_model(train_df, None)
    assert model_info['R2'] > 0.5
    gene_residuals = (residuals.groupby('target_gene')
                      .agg({'residual_z': 'mean'})
                      .sort_values('residual_z')
                      .reset_index())
    assert gene_residuals.loc[0, 'target_gene'] == 'MCL1'
    assert gene_residuals['target_gene'].iloc[-1] == 'BCL2L1'
    ctl_model_info, _ = score.fit_anchor_model(train_df, ['CD81', 'HPRT intron'])
    assert model_info['f_pvalue'] < ctl_model_info['f_pvalue']


def test_get_residual(bigpapi_lfcs):
    model_info_df, guide_residuals = gnt.get_residuals(bigpapi_lfcs, ['CD81', 'HPRT intron'])
    assert ((guide_residuals.groupby(['anchor_gene', 'target_gene', 'condition'])
             .agg({'residual_z': 'mean'})
             .sort_values('residual_z')
             .reset_index()
             .head(1)
             [['anchor_gene', 'target_gene']]
             .values) == [['MAPK3', 'MAPK1']]).all()


def test_order_genes():
    genes = pd.DataFrame({'anchor_gene': ['A', 'B', 'B'], 'target_gene': ['B', 'B', 'A']})
    ordered_genes = score.order_genes(genes)
    expected_order = genes.copy()
    expected_order['gene_a'] = ['A', 'B', 'A']
    expected_order['gene_b'] = ['B', 'B', 'B']
    pd.testing.assert_frame_equal(ordered_genes, expected_order)


def test_get_gene_results(bigpapi_lfcs):
    model_info_df, guide_residuals = gnt.get_residuals(bigpapi_lfcs, ['CD81', 'HPRT intron'])
    gene_results = gnt.get_gene_results(guide_residuals)
    assert ((gene_results.sort_values('z_score')
             .head(1)
             [['gene_a', 'gene_b']]
             .values) == [['MAPK1', 'MAPK3']]).all()

