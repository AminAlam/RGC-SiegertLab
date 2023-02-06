import click
import os
from bin_conversion import *


@click.group(chain=True)
@click.option('--sorting_file_directory', default='/mnt/hdd1/RGC/sorting_files')
@click.option('--h5_files_directory', default='/mnt/hdd1/RGC/data')
@click.option('--stimulus_file_directory', default='/mnt/hdd1/RGC/data/stim')
@click.option('--bin_files_directory', default='/mnt/hdd2/RGC/bin_data')
@click.option('--routes')
@click.pass_context
def cli(ctx, sorting_file_directory, h5_files_directory, stimulus_file_directory, bin_files_directory, routes):
    routes = routes.strip('[]').split(',')
    ctx.obj = {'sorting_file_directory': sorting_file_directory, 
                'h5_files_directory': h5_files_directory, 
                'stimulus_file_directory':stimulus_file_directory, 
                'bin_files_directory':bin_files_directory,
                'routes':routes}

@cli.command('binary_conversion')
@click.pass_context
def binary_conversion(ctx):
    print('bin conversion')
    binary_conversion_backend(ctx)

@cli.command('mea_sort')
@click.option('--kilosort_path')
@click.option('--npy_matlab_path')
@click.option('--path2cnfg')
@click.option('--temporary_folder_directory')
@click.pass_context
def mea_sort(ctx, kilosort_path, npy_matlab_path, path2cnfg, temporary_folder_directory):
    print('mea sorting')
    routes = ctx.obj['routes']
    bin_files_directory = ctx.obj['bin_files_directory']
    h5_files_directory = ctx.obj['h5_files_directory']
    sorting_file_directory = ctx.obj['sorting_file_directory']
    for route in routes:
        sorting_file_path = os.path.join(sorting_file_directory, route)
        os.system(f"matlab -nodisplay -r \"cd('developing/'); MEA_sort3('{sorting_file_path}', '{bin_files_directory}', '{h5_files_directory}', '{kilosort_path}', '{npy_matlab_path}', '{path2cnfg}', '{temporary_folder_directory}');exit\" ") 
        os.system("cd ..")


if __name__=='__main__':
    cli()