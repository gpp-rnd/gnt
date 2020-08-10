"""Console script for gnt."""
import click
import pandas as pd
import gnt


@click.command()
@click.argument('input_file',
                required=True)
@click.argument('output_base_name')
@click.option('--control', help='Negative control genes to calculate base LFCs', required=True, multiple=True)
@click.option('--score', default='residual', type=click.Choice(['residual', 'dlfc']),
              help='Method for calculating combinatorial interactors')
@click.option('--in_delim', default=',', help='Delimiter to read files.')
@click.option('--fit_genes', default=None, help='Genes used to fit linear models. Defaults to all genes.')
def main(input_file, in_delim, score, output_base_name, control, fit_genes):
    """Calculate Genetic iNTeractions

    INPUT_FILE: Delimited file of log-fold changes from combinatorial screening data. columns should be ordered
    guide 1, guide 2, gene 1, gene 2, conditions...

    OUTPUT_BASE_NAME: Base name of output file

    """
    lfcs = pd.read_csv(input_file, delimiter=in_delim)
    if score == 'residual':
        guide_scores, _ = gnt.get_guide_residuals(lfcs, control, fit_genes)
        gene_scores = gnt.get_gene_residuals(guide_scores, 'residual_z')
    elif score == 'dlfc':
        guide_scores = gnt.get_guide_dlfcs(lfcs, control)
        gene_scores = gnt.get_gene_dlfcs(guide_scores, 'dlfc')
    else:
        raise ValueError('score argument:' + score + ' not recognized')
    guide_scores.to_csv(output_base_name + '_gnt_' + score + '_guide__scores.csv', index=False)
    gene_scores.to_csv(output_base_name + '_gnt_' + score + '_gene_scores.csv',  index=False)


if __name__ == "__main__":
    main()
