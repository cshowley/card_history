import argparse
from pymongo import MongoClient
import constants


def query_predictions(gemrate_id, grade, half_grade, grading_company):
    if not constants.S1_MONGO_URL:
        print("Error: MONGO_URL not set.")
        return

    client = MongoClient(constants.S1_MONGO_URL)
    db = client[constants.S1_DB_NAME]
    collection = db[constants.S10_PREDICTIONS_COLLECTION]

    # MongoDB structure expectation:
    # gemrate_id: str
    # grade: int or float
    # half_grade: int or bool (0 or 1)
    # grading_company: str

    if str(half_grade).lower() in ["true", "1"]:
        half_grade_val = 1
    elif str(half_grade).lower() in ["false", "0"]:
        half_grade_val = 0
    else:
        half_grade_val = half_grade

    query = {
        "gemrate_id": gemrate_id,
        "grade": grade,
        "half_grade": half_grade_val,
        "grading_company": grading_company,
    }

    print(
        f"Connecting to {constants.S1_DB_NAME}.{constants.S10_PREDICTIONS_COLLECTION}"
    )
    print(f"Querying with filter: {query}")

    results = list(collection.find(query))

    if not results:
        print("No document found matching the exact query.")
        if isinstance(grade, int):
            query["grade"] = float(grade)
            print(f"Retrying with grade as float: {query['grade']}")
            results = list(collection.find(query))
        elif isinstance(grade, float) and grade.is_integer():
            query["grade"] = int(grade)
            print(f"Retrying with grade as int: {query['grade']}")
            results = list(collection.find(query))

    if not results:
        print("Still no documents found.")
    else:
        print(f"Found {len(results)} document(s):")
        for doc in results:
            print("---")
            for k, v in doc.items():
                print(f"{k}: {v}")
            print("---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query predictions from MongoDB.")
    parser.add_argument(
        "--gemrate_id",
        type=str,
        required=True,
        help="Gemrate ID (e.g., 'gemrate_id_example')",
    )
    parser.add_argument("--grade", type=float, required=True, help="Grade (e.g., 10)")
    parser.add_argument(
        "--half_grade",
        type=str,
        required=True,
        help="Half Grade (0 or 1, True or False)",
    )
    parser.add_argument(
        "--grading_company",
        type=str,
        required=True,
        help="Grading Company (PSA, BGS, CGC)",
    )

    args = parser.parse_args()

    if args.grade.is_integer():
        grade = int(args.grade)
    else:
        grade = args.grade

    query_predictions(args.gemrate_id, grade, args.half_grade, args.grading_company)
