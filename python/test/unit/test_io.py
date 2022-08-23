import unittest

import arbor as A

from pathlib import Path
from tempfile import TemporaryDirectory as TD
from io import StringIO

acc = """(arbor-component
  (meta-data
    (version "0.1-dev"))
  (cable-cell
    (morphology
      (branch 0 -1
        (segment 0
          (point -3.000000 0.000000 0.000000 3.000000)
          (point 3.000000 0.000000 0.000000 3.000000)
          1)))
    (label-dict
      (region-def "soma"
        (tag 1))
      (locset-def "mid"
        (location 0 0.5)))
    (decor
      (default
        (membrane-potential -40.000000))
      (default
        (ion-internal-concentration "ca" 0.000050))
      (default
        (ion-external-concentration "ca" 2.000000))
      (default
        (ion-reversal-potential "ca" 132.457934))
      (default
        (ion-internal-concentration "k" 54.400000))
      (default
        (ion-external-concentration "k" 2.500000))
      (default
        (ion-reversal-potential "k" -77.000000))
      (default
        (ion-internal-concentration "na" 10.000000))
      (default
        (ion-external-concentration "na" 140.000000))
      (default
        (ion-reversal-potential "na" 50.000000))
      (paint
        (tag 1)
        (density
          (mechanism "default::hh"
            ("gnabar" 0.120000)
            ("el" -54.300000)
            ("q10" 0.000000)
            ("gl" 0.000300)
            ("gkbar" 0.036000))))
      (place
        (location 0 0.5)
        (current-clamp
          (envelope
            (10.000000 0.800000)
            (12.000000 0.000000))
          0.000000 0.000000)
        "I Clamp 0"))))
"""

class TestAccIo(unittest.TestCase):
    def test_stringio(self):
        sio = StringIO(acc)
        A.load_component(sio)

    def test_fileio(self):
        fn = 'test.acc'
        with TD() as tmp:
            tmp = Path(tmp)
            with open(tmp / fn, 'w') as fd:
                fd.write(acc)
            with open(tmp / fn) as fd:
                A.load_component(fd)

    def test_nameio(self):
        fn = 'test.acc'
        with TD() as tmp:
            tmp = Path(tmp)
            with open(tmp / fn, 'w') as fd:
                fd.write(acc)
            A.load_component(str(tmp / fn))

    def test_pathio(self):
        fn = 'test.acc'
        with TD() as tmp:
            tmp = Path(tmp)
            with open(tmp / fn, 'w') as fd:
                fd.write(acc)
            A.load_component(tmp / fn)
