import os
import boto3
from botocore.handlers import disable_signing
import tarfile
import os
import time
t1=time.time()

os.chdir('/home/sevir_data')

data_path='/home/sevir_data'
DEST_TRAIN_FILE= os.path.join(data_path,'data/processed/nowcast_training_000.h5')
DEST_TRAIN_META=os.path.join(data_path, 'data/processed/nowcast_training_000_META.csv')
DEST_TEST_FILE= os.path.join(data_path, 'data/processed/nowcast_testing_000.h5')
DEST_TEST_META= os.path.join(data_path, 'data/processed/nowcast_testing_000_META.csv')

resource = boto3.resource('s3')
resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
bucket=resource.Bucket('sevir')

print('Dowloading sample training data')
if not os.path.exists(DEST_TRAIN_FILE):
    bucket.download_file('data/processed/nowcast_training_000.h5.tar.gz',DEST_TRAIN_FILE+'.tar.gz')
    bucket.download_file('data/processed/nowcast_training_000_META.csv',DEST_TRAIN_META)
    with tarfile.open(DEST_TRAIN_FILE+'.tar.gz') as tfile:
        tfile.extract('data/processed/nowcast_training_000.h5','..')
else:
    print('Train file %s already exists' % DEST_TRAIN_FILE)
print('Dowloading sample testing data')
if not os.path.exists(DEST_TEST_FILE):
    bucket.download_file('data/processed/nowcast_testing_000.h5.tar.gz',DEST_TEST_FILE+'.tar.gz')
    bucket.download_file('data/processed/nowcast_testing_000_META.csv',DEST_TEST_META)
    with tarfile.open(DEST_TEST_FILE+'.tar.gz') as tfile:
        tfile.extract('data/processed/nowcast_testing_000.h5','..')
else:
    print('Test file %s already exists' % DEST_TEST_FILE)

print('Done downloading')
t2=time.time()
print('total downloading time is ',(t2-t1))