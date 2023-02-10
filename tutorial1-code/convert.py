import csv
import json

simple_labels = ["agreement", "elaboration"]


def main(simple: bool):
    with open(f"data/coarse_discourse_dataset{'.simple' if simple else ''}.csv", "w", newline='') as csvfile:
        out = csv.DictWriter(csvfile, dialect="unix", fieldnames=["id", "annot1", "annot2", "annot3"],
                             extrasaction="ignore")
        out.writeheader()
        with open("data/coarse_discourse_dataset.json") as fin:
            for line in fin.readlines():
                record = json.loads(line)

                for post in record["posts"]:
                    if len(post["annotations"]) == 3:
                        item = {
                            "id": post["id"],
                            "annot1": post["annotations"][0]["main_type"],
                            "annot2": post["annotations"][1]["main_type"],
                            "annot3": post["annotations"][2]["main_type"],
                        }
                        print(item)
                        # binary case, simplified code
                        if simple:
                            if item["annot1"] in simple_labels and item["annot2"] in simple_labels and item[
                                "annot3"] in simple_labels:
                                out.writerow(item)
                        else:
                            out.writerow(item)


if __name__ == "__main__":
    main(simple=False)
    main(simple=True)
