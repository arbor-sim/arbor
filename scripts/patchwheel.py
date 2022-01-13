import shutil,subprocess,argparse
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Patch Arbor wheels build with skbuild and correct by auditwheel. Linux only.')
    parser.add_argument('path', type=dir_path, help='The path in which the wheels will be patched.')
    parser.add_argument('-ko','--keepold', action='store_true', help='If you want to keep the old wheels in /old')

    return parser.parse_args()

def dir_path(path):
    path = Path(path)
    if Path.is_dir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

parsed_args = parse_arguments()
Path.mkdir(parsed_args.path / 'old', exist_ok=True)

for inwheel in parsed_args.path.glob("*.whl"):
    zipdir = Path(f"{inwheel}.unzip")
    shutil.unpack_archive(inwheel,zipdir,'zip')

    arborn = list(zipdir.glob("**/_arbor.cpython*.so"))[0]
    libxml2n = list(zipdir.glob("**/libxml2*.so*"))[0]
    try:
        subprocess.check_call(f"patchelf --set-rpath '$ORIGIN/../arbor.libs' {arborn}",shell=True)
    except subprocess.CalledProcessError as e:
        print(f"shit hit the fan executing patchelf on {arborn}")
    try:
        subprocess.check_call(f"patchelf --set-rpath '$ORIGIN' {libxml2n}",shell=True)
    except subprocess.CalledProcessError as e:
        print(f"shit hit the fan executing patchelf on {libxml2n}")

    # TODO? correct checksum/bytecounts in *.dist-info/RECORD.
    # So far, Python does not report mismatches

    outwheel = Path(shutil.make_archive(inwheel, 'zip', zipdir))
    Path.rename(inwheel, parsed_args.path / 'old' / inwheel.name)
    Path.rename(outwheel, parsed_args.path / inwheel.name)

if not parsed_args.keepold:
    Path.rmdir(parsed_args.path / 'old')