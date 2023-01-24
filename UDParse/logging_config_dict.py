cfg = { "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(levelname)s %(name)s %(asctime)s %(message)s",
                },
#            "flaskfmt": {
#                "format": "%(levelname)s dddddd %(name)s ttttt %(asctime)s sssss %(message)s",
#                },
            
            },
        "handlers": { "default": { "class": "logging.StreamHandler", "formatter": "default", },
#                      "flaskhandler": { "class": "logging.StreamHandler", "formatter": "flaskfmt", },
                      },

        "loggers": { "udparse": { "handlers": ["default"], "level": "WARN", "propagate": False},
#                     "werkzeug": { "handlers": ["flaskhandler"], "level": "DEBUG", "propagate": False}
                     },
        }
