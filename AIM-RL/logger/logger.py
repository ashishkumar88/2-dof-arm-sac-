#!/usr/bin/env python

# Copyright 2023 OffWorld Inc.
# Doing business as Off-World AI, Inc. in California.
# All information, including without limitation, any
# source code, images, and data, included in this 
# repository is proprietary and confidential information 
# of Off-World, Inc., is subject to the terms of confidentiality
# arrangements, and may not be distributed to or shared 
# with any third parties without the explicit written 
# consent of Off-World, Inc.

"""Creates a custom logger object
"""

import logging
import sys

# Import from Jack's work
level = logging.INFO

class LogColorFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def create_logger(process_name):
    log = logging.getLogger(process_name)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(LogColorFormatter())
    log.addHandler(handler)
    log.setLevel(level)
    log.propagate = False
    return log
