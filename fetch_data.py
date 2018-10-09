import requests
import os
import zipfile


data_dir = os.getcwd()
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

#Train data
image_zip_file_train = 'dataset1.zip'
file_id_train = "0B0d9ZiqAgFkiOHR1NTJhWVJMNEU"
destination_train = 'data.zip'



def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

#  TODO Add progress bar
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def extract_zip_file(zip_file_name, destination):
    zip_ref = zipfile.ZipFile(zip_file_name, 'r')
    zip_ref.extractall(destination)
    zip_ref.close()


def download_data(train=True,data_dir=data_dir,
            file_id_train=file_id_train,destination_train=destination_train,image_zip_file_train=image_zip_file_train):

    if train:
        print('downloading the train images from google drive...')
        download_file_from_google_drive(file_id_train, destination_train)
        extract_zip_file(os.path.join(data_dir, image_zip_file_train), data_dir)
        os.remove(os.path.join(data_dir,image_zip_file_train))
        
        

    print("done !")
if __name__ == "__main__":
    

    download_data(train=True)

    
