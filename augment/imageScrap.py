import urllib

def scrap(base_url, image_dest, image_extension, image_range):
    i = 1
    c = 1
    while i <= image_range:
        file = base_url + str(i) + image_extension
        file_name = file.split("/")[-1]
        file_desc = image_dest + file_name
        try:
            urllib.urlretrieve(file, file_desc)
            print("Image Downloaded : {0} / {1} ".format(c, image_range))
            c +=1
        except Exception:
            print("Error While Downloading : {0}".format(file_name))
            continue
        i += 1


def main():
    base_url = "http://www.dermnet.com/dn2/allJPG3/malignant-melanoma-"
    image_dest = "/home/openroot/Tanmoy/Working Stuffs/myStuffs/havss-tf/ISIC-2017/data/scrap/"
    image_extension = ".jpg"
    scrap(base_url, image_dest, image_extension, 191)

if __name__ == '__main__':
    main()