import os
import csv
import yaml


class CSVLogger:
    '''
    data logger for recording logging data and saving them as a .csv file
    '''
    def __init__(self, file_path, columns_names=None, append=False):
        assert os.path.splitext(file_path)[1].lower() == '.csv', "Invalid file name for creating .csv file."
        self.file_path = file_path
        self.columns = columns_names
        if self.file_exist():
            if append == False:
                raise FileExistsError("CSV logging file to be created already exists and appending is disabled.")
        else:
            if columns_names is not None:
                self.add_row_from_list(columns_names)

    def file_exist(self):
        return os.path.exists(self.file_path)

    def add_row_from_list(self, list_data):
        assert isinstance(list_data, list), "Invalid type for input row data, which is supposed to be in list type."
        with open(self.file_path, 'a', newline='') as f:
            csv.writer(f).writerow(list_data)

    def add_row_from_dict(self, dict_data):
        assert isinstance(dict_data, dict), "Invalid type for input row data, which is supposed to be in dict type."
        assert all([k in self.columns for k in dict_data.keys()]), \
            "One or more keys in the dictionary do not match any column name in pre-defined data logger."
        list_data = [dict_data[k] for k in self.columns]
        with open(self.file_path, 'a', newline='') as f:
            csv.writer(f).writerow(list_data)

    def _clear_content(self):
        f = open(self.file_path, "w")
        f.truncate()
        f.close()

    def _read_rows(self):
        rows = []
        with open(self.file_path) as f:
            for r in csv.reader(f):
                rows.append(r)
        return rows

    def add_column(self, head, values):
        assert isinstance(head, str), "Invalid type for column head, which is supposed to be in str type."
        assert isinstance(values, list), "Invalid type for column values, which is supposed to be in list type."

        if self.columns is not None:
            old_rows = self._read_rows()
            assert len(old_rows) == (len(values) + 1), \
                "Fail to append column: length of the values is inconsistent with the existing number of rows!"
            new_rows = []
            index = 0
            for r in old_rows:
                if r == self.columns:
                    new_rows.append(r + [head])
                else:
                    new_rows.append(r + [values[index]])
                    index += 1
            self.columns.append(head)
        else:
            new_rows = [[head]] + [[v] for v in values]
            self.columns = [head]

        self._clear_content()

        with open(self.file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for row in new_rows:
                writer.writerow(row)


# check and create directory
def make_dir(dir_path, allow_repeat=False):
    if allow_repeat:
        index = 1
        while True:
            new_path = os.path.join(os.path.dirname(dir_path), '{}_{}'.format(os.path.basename(dir_path), index))
            if os.path.exists(new_path):
                index += 1
                continue
            else:
                os.mkdir(new_path)
                return new_path
    else:
        if not(os.path.exists(dir_path)):
            os.mkdir(dir_path)
        return dir_path


# save configurations to a yaml file in the specific path
def save_config(config, path, file_name='config.yaml'):
    file_path = os.path.join(path, file_name)
    with open(file_path, 'w', encoding = 'utf-8') as f:
        yaml.dump(config, f)
    return file_path