def write_log(log, log_path):
    print(log)
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()