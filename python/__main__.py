import os, sys, subprocess

if len(sys.argv) < 2:
    print('No argument provided for arbor module.', file=sys.stderr)
    sys.exit(1)

if sys.argv[1] == 'modcc':
    sys.exit(subprocess.call([
        os.path.join(os.path.dirname(__file__), "bin", "modcc"),
        *sys.argv[2:]
    ]))
if sys.argv[1] == 'build-catalogue' or sys.argv[1] == 'arbor-build-catalogue':
    sys.exit(subprocess.call([
        os.path.join(os.path.dirname(__file__), "bin", "arbor-build-catalogue"),
        *sys.argv[2:]
    ]))
else:
    print('Unknown argument: ' + sys.argv[1], file=sys.stderr)
    sys.exit(1)
