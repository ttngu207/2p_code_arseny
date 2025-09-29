import os
from pathlib import Path
from datetime import datetime, timezone
import shutil
from tqdm import tqdm
import datajoint as dj
import numpy as np

from pipeline_py.utils import nwb_export


logger = dj.logger

lab = dj.VirtualModule('lab', 'map_lab')
experiment = dj.VirtualModule('experiment', 'arseny_s1alm_experiment2')
imaging = dj.VirtualModule('imaging', 'arseny_learning_imaging')
stim_anal = dj.VirtualModule('stim_anal', 'arseny_learning_photostim_anal')
paper = dj.VirtualModule('paper', 'arseny_learning_photostim_paper')

schema = dj.Schema("arseny_export") 

nwb_root_dir = Path(dj.config["custom"].get("nwb_root_dir", "."))
nwb_dir = nwb_root_dir / "export" / "2p_code_arseny_paper2025" / "NWB"


@schema
class NWBFileExport(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    execution_time: datetime
    execution_duration: float  # in hours
    nwb_filepath: varchar(255) # relative path to nwb file
    file_size: float  # in bytes
    """

    @property
    def key_source(self):
        sessions_with_both_photostim_and_behavior = (
            experiment.Session & stim_anal.SessionEpochsIncludedFinalUniqueEpochs & paper.ROILICK2DInclusion
        )
        sessions_with_photostim_only = (
            experiment.Session & stim_anal.SessionEpochsIncludedFinalUniqueEpochs - paper.ROILICK2DInclusion
        )
        sessions_without_photostim = (
            experiment.Session & paper.ROILICK2DInclusion - stim_anal.SessionEpochsIncludedFinalUniqueEpochs
        )
        return sessions_with_both_photostim_and_behavior.proj() + sessions_with_photostim_only.proj() + sessions_without_photostim.proj()
    
    def make(self, key):
        execution_time = datetime.now(timezone.utc)

        nwb_dir.mkdir(exist_ok=True, parents=True)
        nwb_filepath = nwb_dir / f'{key["subject_id"]}_{key["session"]}.nwb'

        nwbfile = nwb_export.session_to_nwb(key)
        nwbfile = nwb_export.add_ophys_plane_to_nwb(key, nwbfile)

        nwb_export.write_nwb(nwbfile, nwb_filepath, validate=True, check_read=True)

        execution_duration = (
            datetime.now(timezone.utc) - execution_time
        ).total_seconds() / 3600

        self.insert1(
            dict(
                key,
                execution_time=execution_time,
                execution_duration=execution_duration,
                nwb_filepath=nwb_filepath.relative_to(nwb_root_dir).as_posix(),
                file_size=nwb_filepath.stat().st_size,
            ),
        )


def perform_dandi_upload(nwb_dir, verify_upload=True):
    """
    Uploads NWB files to DANDI archive and optionally verifies the upload.

    This function handles the entire DANDI upload process including:
    1. Setting up a temporary DANDI directory
    2. Uploading NWB files using the DANDI API
    3. Optionally verifying that files were uploaded correctly by comparing file sizes

    Args:
        nwb_dir (Path): Directory containing NWB files to upload
        verify_upload (bool, optional): Whether to verify the upload by checking file sizes. Defaults to True.

    Raises:
        Exception: If DANDISET_ID or DANDI_API_KEY environment variables are not set
        Exception: If file size verification fails for any uploaded file

    Note:
        Requires DANDISET_ID and DANDI_API_KEY to be set either in environment variables
        or in datajoint config under 'custom' section.
    """
    from element_interface.dandi import upload_to_dandi
    from dandi import upload as dandi_upload, exceptions as dandi_exceptions
    from dandi.dandiapi import DandiAPIClient

    start_time = datetime.now(timezone.utc)

    dandiset_id = os.getenv("DANDISET_ID", dj.config["custom"].get("DANDISET_ID"))
    dandi_api_key = os.getenv("DANDI_API_KEY", dj.config["custom"].get("DANDI_API_KEY"))
    if not dandiset_id or not dandi_api_key:
        raise Exception("DANDISET_ID and DANDI_API_KEY must be set in the environment")

    dandiset_dir = nwb_dir.parent / "DANDI"
    if dandiset_dir.exists():
        shutil.rmtree(dandiset_dir)

    dandiset_dir.mkdir(parents=True, exist_ok=True)

    upload_to_dandi(
        data_directory=nwb_dir,
        dandiset_id=dandiset_id,
        staging=False,
        working_directory=dandiset_dir,
        api_key=dandi_api_key,
        sync=False,
        existing=dandi_upload.UploadExisting.OVERWRITE,
        shell=False,
    )

    if verify_upload:
        nwb_files = list(nwb_dir.rglob("*.nwb"))
        for nwb_filepath in tqdm(nwb_files, desc="Verifying DANDI upload"):
            subject_name, session_id = (
                NWBFileExport
                & {"nwb_filepath": nwb_filepath.relative_to(nwb_root_dir).as_posix()}
            ).fetch1("subject_id", "session")

            remote_path = next(
                dandiset_dir.rglob(
                    f"{dandiset_id}/sub-{subject_name}/*ses-{session_id}*.nwb"
                )
            )
            with dandi_upload.ExitStack() as stack:
                # We need to use the client as a context manager in order to ensure the
                # session gets properly closed.  Otherwise, pytest sometimes complains
                # under obscure conditions.
                client = stack.enter_context(DandiAPIClient.for_dandi_instance("dandi"))
                client.check_schema_version()
                client.dandi_authenticate()

                remote_dandiset = client.get_dandiset(dandiset_id, "draft")
                try:
                    extant = remote_dandiset.get_asset_by_path(
                        f"sub-{subject_name}/{remote_path.name}"
                    )
                except dandi_exceptions.NotFoundError:
                    remote_filesize = 0
                else:
                    remote_filesize = extant.size

            if remote_filesize != nwb_filepath.stat().st_size:
                raise Exception(
                    f"DANDI upload failed for {nwb_filepath}\n\tExpected size: {nwb_filepath.stat().st_size}\n\tActual size: {remote_filesize}"
                )

    logger.info(f"DANDI upload completed in {datetime.now(timezone.utc) - start_time}")

