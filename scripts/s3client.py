import boto3
import dotenv

class S3Client:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=dotenv.get_key(".aws.env", "AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=dotenv.get_key(".aws.env", "AWS_SECRET_ACCESS_KEY"),
            aws_session_token=dotenv.get_key(".aws.env", "AWS_SESSION_TOKEN"),
            region_name=dotenv.get_key(".aws.env", "REGION_NAME")
        )

    def upload(self, locals, bucket, remotes):
        for local_file, remote_file in zip(locals, remotes):
            self.s3.upload_file(local_file, bucket, remote_file)

    def exists(self, filepath, bucket):
        try:
            self.s3.head_object(Bucket=bucket, Key=filepath)
            return True
        except self.s3.exceptions.ClientError as e:
            return False
        
    def download(self, remotes, bucket, locals):
        for remote_file, local_file in zip(remotes, locals):
            self.s3.download_file(bucket, remote_file, local_file)