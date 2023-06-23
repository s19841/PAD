import csv



def format_to_csv(file_in, file_out):
    file1 = open(file_in, 'r')
    Lines = file1.readlines()
    lines_formated = []
    count = 0

    formated_line = ""
    for line in Lines:
        count += 1
        formated_line += " " + line.strip('\n')
        if count == 10:
            count = 0
            lines_formated.append(formated_line[1:])
            formated_line = ""

    with open(file_out, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for l in lines_formated:
            writer.writerow([l])


def format_files():
    format_to_csv('data/cleveland.data',
                  'data_formated/cleveland.csv')
    format_to_csv('data/hungarian.data',
                  'data_formated/hungarian.csv')
    format_to_csv('data/switzerland.data',
                  'data_formated/switzerland.csv')
    format_to_csv('data/long-beach-va.data',
                  'data_formated/long-beach-va.csv')


if __name__ == '__main__':
    format_files()
