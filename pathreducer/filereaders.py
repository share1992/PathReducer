# MIT License
#
# Copyright (c) 2019 Lars Andersen Bratholm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import glob

class XYZReader(object):
    """
    File reader for XYZ format. Adapted from QML: https://github.com/qmlcode/qml
    """

    def __init__(self, filenames=None):
        """
        :param filenames: single filename, list of filenames or a string to be read by glob. e.g. 'dir/*.xyz'
        :type filenames: list or string
        """

        if isinstance(filenames, str):
            if ".xyz" not in filenames:
                filenames = sorted(glob.glob(filenames))
            else:
                filenames = [filenames]

        self._parse_xyz_files(filenames)

    def get_filenames(self):
        """
        Returns a list of filenames in the order they were parsed.
        """
        return self.filenames

    def _parse_xyz_files(self, filenames):
        """
        Parse a list of xyz files.
        """

        n = len(filenames)

        coordinates = []
        elements = []
        files = []

        # Report if there's any error in reading the xyz files.
        try:
            for i, filename in enumerate(filenames):
                with open(filename, "r") as f:
                    lines = f.readlines()

                natoms = int(lines[0])

                if len(lines) % (natoms + 2) != 0:
                    raise SystemExit("1 Error in parsing coordinates in %s" % filename)

                n_snapshots = len(lines) // (natoms + 2)

                if n_snapshots > 1:
                    traj_flag = True
                else:
                    traj_flag = False

                for k in range(n_snapshots):
                    elements.append(np.empty(natoms, dtype='<U3'))
                    coordinates.append(np.empty((natoms, 3), dtype=float))

                    if traj_flag:
                        files.append(filename + "_%d" % k)
                    else:
                        files.append(filename)

                    for j, line in enumerate(lines[k * (natoms+2) + 2: (k+1) * (natoms + 2)]):
                        tokens = line.split()

                        if len(tokens) < 4:
                            raise SystemExit("2 Error in parsing coordinates in %s" % filename)

                        elements[-1][j] = tokens[0]
                        coordinates[-1][j] = np.asarray(tokens[1:4], dtype=float)
        except:
            raise SystemExit("3 Error in reading %s" % filename)

        # Set coordinates and nuclear_charges
        self.coordinates = np.asarray(coordinates)
        self.elements = np.asarray(elements, dtype='<U3')
        self.filenames = files
