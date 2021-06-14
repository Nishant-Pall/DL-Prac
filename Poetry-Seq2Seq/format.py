import sys


def clean_data(file_name, trim_length):
    t = 0
    shortened_txt = []
    file_name = "spa.txt"
    for lines in open(file_name):
        t += 1
        if t > trim_length:
            break
        lines = lines.split('\t')
        shortened_txt.append([lines[0], lines[1]])

    output_file = open(f'{file_name}_cleaned.txt', "w")
    for translation_pair in shortened_txt:
        output_file.write(f'{translation_pair}\n')
    output_file.close()


if len(sys.argv) > 2:
    file_name = sys.argv[1]
    trim_length = sys.argv[2]
    trim_length = int(trim_length)
else:
    file_name = input("Enter name of file to be cleaned: ")
    trim_length = int(input(
        "Enter the length upto which you want to trim the file: "))

clean_data(file_name=file_name, trim_length=trim_length)
