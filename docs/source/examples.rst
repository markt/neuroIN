Examples
=============

Installation/Usage:
*******************
Install with Test PyPi: `pip install -i https://test.pypi.org/simple/ neuroIN`


Import some data and load
**************************************************
.. code-block:: python
    from neuroIN.datasets import eegmmidb
    from neuroIN.io import dataset
    
    eegmmidb.import_eegmmidb('data/eegmmidb')
    data = dataset.Dataset('data/eegmmidb')
