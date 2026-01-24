import numpy as np
import os
import matplotlib.pyplot as plt


src_root1 = '/ailab/data/Cardiac super-resolution label maps/pc1/225'
dst_root = '/ailab/data/Cardiac super-resolution label maps/prints1/'

###  number = '1004' CT LAX has a problem


for subroot, dirs, files in os.walk(src_root1):
    if len(files) < 8: continue

    patient_num_vec = subroot.split(os.path.sep)
    p1 = patient_num_vec[-1]
    print('working on %s', p1)
    for file in files:
        if file.endswith('ES_rv.npz'):
            epi_myo_file = os.path.join(subroot, file.replace('rv', 'epi_myo'))
            endo_myo_file = os.path.join(subroot, file.replace('rv', 'endo_myo'))
            rv_file = os.path.join(subroot, file)
            pc_file = os.path.join(subroot, file.replace('rv', 'pc'))

            npz = np.load(epi_myo_file)
            np_arrs = []
            for np_file in npz.keys(): np_arrs.append(npz[np_file])
            epi_myo = np_arrs[0]

            npz = np.load(endo_myo_file)
            np_arrs = []
            for np_file in npz.keys(): np_arrs.append(npz[np_file])
            endo_myo = np_arrs[0]

            npz = np.load(rv_file)
            np_arrs = []
            for np_file in npz.keys(): np_arrs.append(npz[np_file])
            rv = np_arrs[0]

            npz = np.load(pc_file)
            np_arrs = []
            for np_file in npz.keys(): np_arrs.append(npz[np_file])
            m_pc = np_arrs[0]


            fig = plt.figure(figsize=(9, 9))
            ax = fig.add_subplot(421, projection='3d')
            ax.scatter(m_pc[4000:4400, 0], m_pc[4000:4400, 1], m_pc[4000:4400, 2], s=2, c='green')
            ax.axis()
            ax.view_init(azim=-87, elev=-1)

            ax = fig.add_subplot(422, projection='3d')
            ax.scatter(m_pc[4400:4800, 0], m_pc[4400:4800, 1], m_pc[4400:4800, 2], s=2, c='green')
            ax.axis()
            ax.view_init(azim=-134, elev=1)

            ax = fig.add_subplot(423, projection='3d')
            ax.scatter(m_pc[5200:5600, 0], m_pc[5200:5600, 1], m_pc[5200:5600, 2], s=2, c='green')
            ax.axis()
            ax.view_init(azim=-87, elev=-1)

            ax = fig.add_subplot(424, projection='3d')
            ax.scatter(m_pc[4800:5200, 0], m_pc[4800:5200, 1], m_pc[4800:5200, 2], s=2, c='green')
            ax.axis()
            ax.view_init(azim=-76, elev=5)

            ax = fig.add_subplot(425, projection='3d')
            ax.scatter(m_pc[4000:, 0], m_pc[4000:, 1], m_pc[4000:, 2], s=2, c='green')
            ax.axis()
            ax = fig.add_subplot(426, projection='3d')
            ax.scatter(m_pc[:4000, 0], m_pc[:4000, 1], m_pc[:4000, 2], s=2, c='green')
            ax.axis()
            ax = fig.add_subplot(427, projection='3d')
            ax.scatter(m_pc[4000:, 0], m_pc[4000:, 1], m_pc[4000:, 2], s=2, c='green')
            ax.scatter(m_pc[0:4000:5, 0], m_pc[0:4000:5, 1], m_pc[0:4000:5, 2], s=2, c='red')
            ax.axis()
            ax = fig.add_subplot(428, projection='3d')
            ax.scatter(m_pc[0:-1:5, 0], m_pc[0:-1:5, 1], m_pc[0:-1:5, 2], s=2, c='green')
            ax.scatter(epi_myo[0:-1:5, 0], epi_myo[0:-1:5, 1], epi_myo[0:-1:5, 2], s=2, c='red')
            ax.scatter(endo_myo[0:-1:5, 0], endo_myo[0:-1:5, 1], endo_myo[0:-1:5, 2], s=2, c='blue')
            ax.scatter(rv[0:-1:5, 0], rv[0:-1:5, 1], rv[0:-1:5, 2], s=2, c='yellow')
            ax.axis()
            plt.show()

            # file_name = p1 + '_' + file.replace('_rv.npz', '.png')
            # fig.savefig(os.path.join(dst_root, file_name), bbox_inches='tight', pad_inches=0.0, dpi=100)
            # plt.close()






