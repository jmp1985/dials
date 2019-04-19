from __future__ import absolute_import, division, print_function

import procrunner
from dials.array_family import flex


def test_combining_spots(dials_data, tmpdir):
    images = [
        f.strpath
        for f in dials_data("centroid_test_data").listdir("centroid*.cbf", sort=True)
    ]
    images_1 = images[0 : int(len(images) / 2)]
    images_2 = images[int(len(images) / 2) :]

    result = procrunner.run(
        ["dials.import", "output.experiments=experiments-1.json"] + images_1,
        working_directory=tmpdir.strpath,
    )
    assert not result["exitcode"] and not result["stderr"]
    assert tmpdir.join("experiments-1.json").check()

    result = procrunner.run(
        ["dials.find_spots", "experiments-1.json", "output.reflections=strong-1.mpack"],
        working_directory=tmpdir.strpath,
    )
    assert not result["exitcode"] and not result["stderr"]
    assert tmpdir.join("strong-1.mpack").check()

    result = procrunner.run(
        ["dials.import", "output.experiments=experiments-2.json"] + images_2,
        working_directory=tmpdir.strpath,
    )
    assert not result["exitcode"] and not result["stderr"]
    assert tmpdir.join("experiments-2.json").check()

    result = procrunner.run(
        ["dials.find_spots", "experiments-2.json", "output.reflections=strong-2.mpack"],
        working_directory=tmpdir.strpath,
    )
    assert not result["exitcode"] and not result["stderr"]
    assert tmpdir.join("strong-2.mpack").check()

    result = procrunner.run(
        [
            "dials.combine_found_spots",
            "experiments-1.json",
            "experiments-2.json",
            "strong-1.mpack",
            "strong-2.mpack",
            "output.reflections=combined.mpack",
            "output.experiments=combined.json",
        ],
        working_directory=tmpdir.strpath,
    )
    assert not result["exitcode"] and not result["stderr"]
    assert tmpdir.join("combined.json").check()
    assert tmpdir.join("combined.mpack").check()

    r = flex.reflection_table.from_msgpack_file(tmpdir.join("combined.mpack").strpath)
    assert r["id"].all_eq(0)
