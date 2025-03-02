from utils.data_utils import generate_data


if __name__ == '__main__':
    ge_list, obsv_z, lmark_m = generate_data(10)

    print(ge_list, obsv_z, lmark_m)
