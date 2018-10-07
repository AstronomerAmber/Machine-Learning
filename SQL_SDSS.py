"""
This file directly queries photometric observations associated with SDSS galaxy
spectra which have spectroscopically confirmed redshifts.
"""
from __future__ import print_function, division
import os
import numpy as np
from sklearn.datasets import get_data_home
from .tools import sql_query

SPECCLASS = ['UNKNOWN', 'STAR', 'GALAXY', 'QSO',
             'HIZ_QSO', 'SKY', 'STAR_LATE', 'GAL_EM']

NOBJECTS = 10000

GAL_COLORS_DTYPE = [('u', float),
                    ('g', float),
                    ('r', float),
                    ('i', float),
                    ('z', float),
                    ('specClass', int),
                    ('redshift', float),
                    ('redshift_err', float)]

ARCHIVE_FILE = 'sdss_photoz.npy'

def sdss_galaxy_colors(data_home=None, download_if_missing=True):
    data_home = get_data_home(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    archive_file = os.path.join(data_home, ARCHIVE_FILE)

    query_text = ('\n'.join((
             "SELECT TOP %i" % NOBJECTS,
             "p.u, p.g, p.r, p.i, p.z, s.specClass, s.z, s.zerr",
             "FROM PhotoObj AS p",
             "JOIN SpecObj AS s ON s.bestobjid = p.objid",
             "WHERE ",
             "(specClass = 2 OR specClass = 3 OR specClass = 1)",
             "AND p.u BETWEEN 0 AND 19.6",
             "AND p.g BETWEEN 0 AND 20")))

    if not os.path.exists(archive_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        print("querying for %i objects" % NOBJECTS)
        print(query_text)
        output = sql_query(query_text)
        print("finished.")

        data = np.loadtxt(output, delimiter=',',
                          skiprows=1, dtype=GAL_COLORS_DTYPE)
        np.save(archive_file, data)

    else:
        data = np.load(archive_file)

    return data
