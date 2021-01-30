terr_threshold = 20
rerr_threshold = 20

# file_suffices = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
file_prefix = "chess"
file_suffices = ["office"]

print("Threshold: {}cm/{}deg.".format(terr_threshold, rerr_threshold))

for file_suffix in file_suffices:
    err_cnt = 0
    err_within_thres_cnt = 0

    with open("./err_gt_" + file_prefix + "_"+file_suffix+".txt", "r+") as f:
        # filename rerr terr
        lines = f.readlines()

    for line in lines:
        line = line.split()
        err_cnt += 1

        if float(line[1]) < rerr_threshold and float(line[2]) < terr_threshold:
            err_within_thres_cnt += 1

    print("Pose acc. for "+file_suffix+": ", err_within_thres_cnt / err_cnt)
    print("--------------------------------------\n")
