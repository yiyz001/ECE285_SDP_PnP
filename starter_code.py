from utils.data_utils import generate_data, kitti_data, construct_data_matrix


if __name__ == '__main__':
    # Generate synthesis data
    # ge_list, obsv_z, lmark_m = generate_data(10)

    # Load data from KITTI, we do not have ground-truth this time
    obsv_z, lmark_m = kitti_data()

    # Get P and Omega matrices
    list_P, list_Omega = construct_data_matrix([obsv_z], [lmark_m])

    # Define and solve the PnP problem
