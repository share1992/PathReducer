import sys
from numpy.distutils.core import Extension, setup

__author__ = "Stephanie Hare, Lars Andersen Bratholm"
__copyright__ = "Copyright 2018"
__credits__ = ["Stephanie Hare, Lars Andersen Bratholm"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Stephanie Hare, Lars Andersen Bratholm"
__email__ = "stephanie.hare@bristol.ac.uk"
__status__ = "Beta"
__description__ = "PathReducer"
__url__ = "https://github.com/share1992/pathreducer"


def requirements():
    with open('requirements.txt') as f:
        return [line.rstrip() for line in f]


# use README.md as long description
def readme():
    with open('README.md') as f:
        return f.read()

def setup_pathreducer():

    setup(

        name="pathreducer",
        # metadata
        version=__version__,
        author=__author__,
        author_email=__email__,
        platforms = 'Any',
        description = __description__,
        long_description = readme(),
        keywords = ['Dimensionality reduction', 'Biochemistry', 'Chemistry'],
        classifiers = [],
        url = __url__,
        install_requires = requirements(),

        # set up package contents

        ext_package = 'pathreducer'
)

if __name__ == '__main__':

    setup_pathreducer()
