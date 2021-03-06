from pathlib import Path

# import from app! Modules won't be found otherwise
from app.solver.data.Job import Job


def test_to_json():
    job = Job(max_length=1200, cut_width=5, target_sizes={"300": 4, "200": 3})
    assert job.json() == '{"max_length": 1200, "cut_width": 5, ' \
                         '"target_sizes": {"300": 4, "200": 3}}'


def test_from_json():
    json_file = Path("./tests/res/in/testjob.json")
    assert json_file.exists()

    with open(json_file, "r") as encoded_job:
        job = Job.parse_raw(encoded_job.read())
        assert job.__class__ == Job
        assert len(job) > 0
