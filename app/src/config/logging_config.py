import logging.config

import os
import logging.config
if os.getenv('TESTING'):
    log_file = 'tests.log'
else:
    log_file = '/app/logs/app.log'

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,  # Preserve existing loggers

    # Formatters define the log message format
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },

    # Handlers determine where the log messages go.
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': log_file,       # Log file location
            'mode': 'a',                 # Append mode
            'formatter': 'standard',
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
    },

    # The root logger configuration applies to all loggers that don't have a specific configuration.
    'root': {
        'handlers': ['file', 'console'],
        'level': 'DEBUG',                 # Set minimum log level
    },

    # Optionally, define loggers for specific parts of your app.
    'loggers': {
        'celery': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
        'app': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
