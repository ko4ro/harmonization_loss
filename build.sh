#!/bin/bash
# -*- coding: utf-8 -*-
# build image
docker build -t nvcr_pytorch24.03-py3 \
    --no-cache \
    --file ./Dockerfile \
    ./
