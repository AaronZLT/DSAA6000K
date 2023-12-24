def count_strings_in_file(file_path, string_list):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().lower()
    except FileNotFoundError:
        print(f"{file_path} undefined")
        return
    except IOError as e:
        print(f"Error in reading {e}")
        return

    counts = {}

    for string in string_list:
        counts[string] = content.count(string.lower())

    return counts

def query(file_path, string_list):
    result = count_strings_in_file(file_path, string_list)
    sorted_result = sorted(result.items(), key=lambda item: item[1], reverse=False)

    if sorted_result:
        for string, count in sorted_result:
            print(f"'{string}' appears {count} times.")
    print("")

if __name__ == "__main__":
    wiki2 = f'/hpc2hdd/home/lzhang330/ssd_workspace/datasets/wikitext-2-raw-v1/wikitext-2-raw/wiki.train.raw'
    wiki103 = f'/hpc2hdd/home/lzhang330/ssd_workspace/datasets/wikitext-103-raw-v1/wikitext-103-raw/wiki.train.raw'

    # city
    city = ['Adamstown', 'Gambier', 'Nukutavake','Tonga','Haumaefa']
    country = ['Pitcairn','Tuvalu','Gilbert','Tonga','Samoa','Tokelau']
    language = ['Konkani','Balochi','Belarusian','Xhosa','Mossi','Uyghur','Shona']
    discipline = ["psychology","Botany","Meteorology","Ethnology","Archaeology"]
    sports = ["American football","soccer","football","ice hockey","hockey","baseball","volleyball","fencing"]
    # query(wiki103,city)
    # query(wiki103,country)
    # query(wiki103,language)
    # query(wiki103,discipline)
    query(wiki103,sports)
