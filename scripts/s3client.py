import boto3
import dotenv
from pathlib import Path

class S3Client:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=dotenv.get_key(".aws.env", "AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=dotenv.get_key(".aws.env", "AWS_SECRET_ACCESS_KEY"),
            aws_session_token=dotenv.get_key(".aws.env", "AWS_SESSION_TOKEN"),
            region_name=dotenv.get_key(".aws.env", "REGION_NAME")
        )

    def upload_folder(self, local_folder, bucket):
        print(local_folder.name)
        self.s3.put_object(Bucket=bucket, Key=local_folder.name + "/")
        for local_file in local_folder.iterdir():
            remote_file = local_folder.name + "/" + local_file.name
            self.s3.upload_file(str(local_file), bucket, remote_file)

    def exists(self, filepath, bucket):
        try:
            self.s3.head_object(Bucket=bucket, Key=filepath)
            return True
        except self.s3.exceptions.ClientError as e:
            return False
        
    def download_folder(self, remote_folder, bucket, local_folder):
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=remote_folder)
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    object_key = obj['Key']
                    filepath = local_folder.joinpath(object_key)
                    if object_key.endswith("/"):
                        filepath.mkdir(exist_ok=True, parents=True)
                    else:
                        self.s3.download_file(bucket, object_key, str(filepath))