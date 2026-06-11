#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:14:09 2025

@author: amittone
"""

import os
import click
from google.cloud import storage

@click.command()
@click.option(
    "--bucket-name", "-b",
    required=True,
    help="Name of the GCS bucket (e.g., 'ucl-hip-ct-allen-c9fymx6anq3qumgn')."
)
@click.option(
    "--prefix", "-p",
    default="",
    help="Prefix/path within the bucket to download (e.g., 'biology-allen-mouse-716331/brain/...')."
)
@click.option(
    "--local-dest", "-d",
    default=".",
    help="Local directory to which files will be saved."
)
@click.option(
    "--anonymous/--no-anonymous",
    default=True,
    help="Use an anonymous client (true) or the default authenticated client (false)."
)
def main(bucket_name, prefix, local_dest, anonymous):
    """
    Download files from a public (or optionally private) Google Cloud Storage bucket.
    By default, uses an ANONYMOUS client, which only works if the bucket is publicly readable.
    Use --no-anonymous if you need authentication and have credentials configured.
    """

    if anonymous:
        # Create an anonymous client (works only if the bucket is publicly readable)
        client = storage.Client.create_anonymous_client()
        click.echo("Using anonymous client...")
    else:
        # Use your default credentials (requires 'gcloud auth application-default login' or similar)
        client = storage.Client()
        click.echo("Using default authenticated client...")

    # Get a reference to the bucket
    bucket = client.bucket(bucket_name)

    # List all objects with the specified prefix
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        # Construct a local file path (mirror the directory structure)
        local_path = os.path.join(local_dest, blob.name)
        # Ensure local folders exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        click.echo(f"Downloading: {blob.name}")
        blob.download_to_filename(local_path)

    click.echo("Download complete!")

if __name__ == "__main__":
    """
    Download public data (anonymous client) from a bucket path:

	./Gdown.py \
	  --bucket-name ucl-hip-ct-allen-c9fymx6anq3qumgn \
	  --prefix biology-allen-mouse-716331/brain/2.195um_left-hemisphere_bm18 \
	  --local-dest /path/to/local/folder

    Use your authenticated credentials (if the bucket is not public):

	# First, ensure you have valid credentials (e.g., via 'gcloud auth login' or 
	# 'gcloud auth application-default login').
	./download_gcs.py \
	  --no-anonymous \
	  --bucket-name my-private-bucket \
	  --prefix some/path \
	  --local-dest /path/to/dest
    """
    main()
