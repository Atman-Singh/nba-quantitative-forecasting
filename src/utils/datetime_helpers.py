import datetime as dt
from datetime import datetime

DATE_FORMAT = "%Y%m%d"
TIMESTAMP_FORMAT = r"%Y%m%d%H%M%S"

class DatetimeHelpers:

    @staticmethod
    def _format_date(date: datetime) -> int:
        return int(date.strftime(DATE_FORMAT))

    @staticmethod
    def get_current_date() -> datetime:
        return datetime.today()
    
    @staticmethod
    def get_timestamp() -> str:
        return datetime.now().strftime(TIMESTAMP_FORMAT)

    @staticmethod
    def decrement_date(date: datetime, increment: int = 1) -> datetime:
        if increment < 1:
            print('increment must be greater than 0')
        return date - dt.timedelta(days=increment)
    
    @staticmethod
    def get_date_format() -> str:
        return DATE_FORMAT