import shutil
import subprocess
from pathlib import Path


def main():
    dist_dir = Path('dist')
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir()

    content_dir = Path('content')
    files_in_content_dir = [file for file in content_dir.glob('**/*') if file.is_file()]
    folders_in_content_dir = [file for file in content_dir.glob('**/*') if file.is_dir()]
    _mkdir_in_dist(folders_in_content_dir)

    for file in files_in_content_dir:
        _generate_dist(file)


def _mkdir_in_dist(folders_in_content_dir):
    for folder in folders_in_content_dir:
        target_path = '/'.join(['dist', *folder.parts[1:]])
        target_path = Path(target_path)
        target_path.mkdir()


def _generate_dist(file):
    if _is_markdown(file):
        _marp_og_image(file)
        _marp_html(file)
    else:
        _copy_to_dist(file)


def _is_markdown(file):
    return file.suffix == '.md' or file.suffix == '.MD'


def _marp_html(file):
    target_parts = ['dist', *file.parts[1:-1]]
    slides_filename = file.stem + '.html'
    slides_path = '/'.join([*target_parts, slides_filename])
    command = ['marp', '--no-stdin', str(file), '-o', slides_path]
    print(' '.join(command))
    subprocess.run(command)


def _marp_og_image(file):
    target_parts = ['dist', *file.parts[1:-1]]
    og_image_path = '/'.join([*target_parts, 'og-image.jpg'])
    command = ['marp', str(file), '-o', og_image_path]
    print(' '.join(command))
    subprocess.run(command)


def _copy_to_dist(file):
    if file.name == '.keep':
        return

    target_path = '/'.join(['dist', *file.parts[1:]])
    print(f'cp {str(file)} {target_path}')
    shutil.copyfile(file, target_path)


if __name__ == '__main__':
    main()
