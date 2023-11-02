"""Arguments related to encounter files"""
import argparse


def get_encounter_file_arguments():
    """
    Return arguments for encounter file processing
    Returns:
        (parser): Argument parser for file arguments
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--encounter-file",
        type=str,
        required=True,
        help="The file containing the encounter history",
    )
    parser.add_argument(
        "--patient-id-column",
        type=str,
        default='PatientID',
        help="Column name for patient id",
    )
    parser.add_argument(
        "--icd-10-column",
        type=str,
        default='ICD10CD',
        help="Column name for icd 10 codes",
    )
    parser.add_argument(
        "--contact-date-column",
        type=str,
        default='ContactDTS',
        help="Column name for contact date",
    )
    parser.add_argument(
        "--sample-result-date-column",
        type=str,
        default='TestDate',
        help="Column name for sample result date",
    )
    parser.add_argument(
        "--time-difference-column",
        type=str,
        default='time_difference',
        help="Column name for tracking time difference between sample and encounters",
    )
    parser.add_argument(
        "--position-column",
        type=str,
        default='position',
        help="Column name for storing the relative positions of icd codes with respect to the current sample",
    )
    parser.add_argument(
        "--past-time-delta",
        type=str,
        default=None,
        help="Past time difference of encounter and sample.",
    )
    parser.add_argument(
        "--future-time-delta",
        type=str,
        default=None,
        help="Future time difference of encounter and sample.",
    )
    parser.add_argument(
        "--use_log_position",
        default=False,
        action="store_true",
        help="Represent time differences in terms of log of days"
    )
    return parser
