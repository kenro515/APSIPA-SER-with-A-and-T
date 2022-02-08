import io
import glob

if __name__ == '__main__':
    in_path = "***"
    out_path = "***"

    paths_train = sorted(
        glob.glob(in_path))
    fw = open(out_path, 'w')

    for path in paths_train:
        with io.open(path, 'r', encoding="utf-8") as fr:
            text = fr.readline()

            text = text.replace('\t', " ")
            text = text.replace('\n', " ")

            emo_label = path.split('/')[6]
            index = path.split('/')[7].split('_')[1].split('.')[0]
            if emo_label == 'ang':
                text = index + '\t' + text + '\t' + '0' + '\t' + '\n'
            elif emo_label == 'joy':
                text = index + '\t' + text + '\t' + '1' + '\t' + '\n'
            elif emo_label == 'sad':
                text = index + '\t' + text + '\t' + '2' + '\t' + '\n'
            else:
                text = index + '\t' + text + '\t' + '3' + '\t' + '\n'
            fw.write(text)
    fw.close()
