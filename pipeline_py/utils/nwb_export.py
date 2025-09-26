import datajoint as dj
import os
from datetime import datetime, timezone
import json
from tqdm import tqdm
import numpy as np
import pynwb
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ophys import (
    Fluorescence,
    ImageSegmentation,
    OpticalChannel,
    RoiResponseSeries,
    TwoPhotonSeries,
)
from pynwb.behavior import Position, SpatialSeries

logger = dj.logger

lab = dj.VirtualModule('lab', 'map_lab')
experiment = dj.VirtualModule('experiment', 'arseny_s1alm_experiment2')
imaging = dj.VirtualModule('imaging', 'arseny_learning_imaging')
stim_anal = dj.VirtualModule('stim_anal', 'arseny_learning_photostim_anal')
paper = dj.VirtualModule('paper', 'arseny_learning_photostim_paper')

# ============================== SET CONSTANTS ==========================================
default_nwb_output_dir = os.path.join('.', 'nwb_data')
zero_zero_time = datetime.strptime('00:00:00', '%H:%M:%S').time()  # no precise time available
institution = 'Janelia Research Campus'
excitation_lambda = 920.0

study_description = dict(
    related_publications='https://doi.org/10.1101/2023.11.25.568673',
    experiment_description='Calcium imaging and targeted optogenetic stimulation of the anterior lateral motor cortex during multi-directional tongue-reaching behavior',
    keywords=['learning', 'motor cortex', 'calcium imaging', 'functional connectivity'])


def session_to_nwb(session_key):
    logger.info(f'Exporting to NWB 2.0 for session: {session_key}...')

    this_session = (experiment.Session * experiment.SessionID & session_key).fetch1()

    session_details = (experiment.SessionTask * experiment.Task
                       * experiment.SessionTraining & session_key)
    if session_details:
        session_details = session_details.fetch1()
    else:
        session_details = {}

    # -- NWB file - a NWB2.0 file for each session
    nwbfile = NWBFile(identifier=f'{this_session["subject_id"]}_session_{this_session["session"]}',
                      session_description=json.dumps(session_details),
                      session_start_time=datetime.combine(this_session['session_date'], zero_zero_time).astimezone(timezone.utc),
                      file_create_date=datetime.now(timezone.utc),
                      experimenter=[this_session['username']],
                      institution=institution,
                      experiment_description=study_description['experiment_description'],
                      related_publications=study_description['related_publications'],
                      keywords=study_description['keywords'])

    # -- subject
    subj = (lab.Subject & session_key).fetch1()
    nwbfile.subject = pynwb.file.Subject(
        subject_id=str(subj['subject_id']),
        description=f'source: {subj["animal_source"]}; cage_number: {subj["cage_number"]}',
        genotype=' x '.join((lab.Subject.GeneModification & subj).fetch('gene_modification')),
        sex=subj['sex'],
        species='Mus musculus',
        date_of_birth=datetime.combine(subj['date_of_birth'], zero_zero_time) if subj['date_of_birth'] else None)

    return nwbfile


def add_ophys_plane_to_nwb(session_key, nwbfile) -> NWBFile:
    """Adds metadata for a scan from database.

    Args:
        session_key (dict): key from Session table
        nwbfile (NWBFile): nwb file
    """
    if not (imaging.Plane & session_key):
        raise ValueError("No image segmentation results found for this session.")

    device = nwbfile.create_device(
        name="Thorlabs Microscope",
        description="Imaging was done with a 920-nm pulsed laser (Chameleon Ultra II, Coherent) and a galvo-resonant scanner",
        manufacturer="Thorlabs",
    )

    plane_query = imaging.FOV * imaging.Plane * imaging.PlaneCoordinates * imaging.PlaneDirectory & session_key
    frame_timestamps = (imaging.FrameTime & session_key).fetch1("frame_timestamps").flatten()
    imaging_frame_rate_volume = (imaging.FOVEpoch & session_key).fetch("imaging_frame_rate_volume", limit=1)[0]
    imaging_frame_rate_plane = (imaging.FOVEpoch & session_key).fetch("imaging_frame_rate_plane", limit=1)[0]

    optical_channels = {channel: OpticalChannel(
        name=f"OpticalChannel{channel}",
        description=f"Optical channel number {channel}",
        emission_lambda=np.nan,
    ) for channel in set(plane_query.fetch("channel_num"))}

    scan_location = (lab.SurgeryLocation * lab.BrainArea & session_key).fetch1()
    scan_location_str = json.dumps(scan_location)

    ophys_module = nwbfile.create_processing_module(
        name="ophys", description="optical physiology processed data"
    )

    for plane_key in (imaging.Plane & session_key).fetch("KEY", order_by="fov_num, channel_num, plane_num"):
        plane_name = f"FOV{plane_key['fov_num']}_pln{plane_key['plane_num']}_chn{plane_key['channel_num']}"

        logger.info(f"Adding plane: {plane_name}")
        optical_channel = optical_channels[plane_key["channel_num"]]

        # Imaging plane
        imaging_plane = nwbfile.create_imaging_plane(
            name=f"ImagingPlane_{plane_name}",
            optical_channel=optical_channel,
            imaging_rate=imaging_frame_rate_volume,
            description=f"Imaging plane for {plane_name}",
            device=device,
            excitation_lambda=excitation_lambda,
            indicator="unknown",
            location=scan_location_str,
            grid_spacing=(1, 1),
            grid_spacing_unit="pixels",
            origin_coords=(0, 0, 0),
            origin_coords_unit="pixels",
        )
        # Two photon series
        dimension = (imaging.FOV & plane_key).fetch1("fov_y_size", "fov_x_size")
        imaging_files = (imaging.PlaneDirectory & plane_key).fetch("local_path_plane_registered")
        two_p_series = TwoPhotonSeries(
            name=f"TwoPhotonSeries_{plane_name}",
            dimension=dimension,
            external_file=imaging_files.tolist(),
            imaging_plane=imaging_plane,
            starting_frame=[0],
            format="external",
            timestamps=frame_timestamps,
        )
        # nwbfile.add_acquisition(two_p_series)

        # Plane Segmentation
        img_seg = ImageSegmentation(name=f"ImageSegmentation_{plane_name}")
        ps = img_seg.create_plane_segmentation(
            name=f"PlaneSegmentation_{plane_name}",
            description="output from segmenting",
            imaging_plane=imaging_plane,
        )
        ophys_module.add(img_seg)

        # Add ROIs to plane segmentation
        roi_query = imaging.ROI & imaging.ROIInclude & plane_key
        roi_keys = roi_query.fetch("KEY", order_by="roi_number")

        logger.info(f"Adding {len(roi_keys)} ROIs for plane: {plane_name}")

        for roi_key in roi_keys:
            x, y, weight = (imaging.ROI & roi_key).fetch1("roi_x_pix", "roi_y_pix", "roi_pixel_weight")
            ps.add_roi(
                id=roi_key["roi_number"],
                pixel_mask=np.asarray(
                    (x.flatten(), y.flatten(), weight.flatten())
                ).T
            )

        rt_region = ps.create_roi_table_region(
            region=list(range(len(roi_keys))),
            description="All ROIs from database.",
        )

        # Add ROI response series
        ROI_TRACE_TABLES = [
            ("ROITrace", "f_trace"), 
            ("ROISpikes", "spikes_trace"), 
            # ("ROITraceNeuropil", "f_trace"), 
            # ("ROIdeltaF", "dff_trace"),
        ]

        fluorescence_series = []
        for tbl_name, attr_name in ROI_TRACE_TABLES:
            tbl = getattr(imaging, tbl_name)
            name = tbl.__name__ + ":" + attr_name

            logger.info(f"Adding {tbl.__name__} ({attr_name}) for plane {plane_key['plane_num']}")
            # query, reshape, and aggregate traces across SessionEpoch
            roi_traces = []
            for roi_key in tqdm(roi_keys):
                traces = (tbl & roi_key).fetch(attr_name, order_by="session_epoch_number")
                concat_traces = np.concatenate([t.flatten() for t in traces])
                roi_traces.append(concat_traces)
                            
            fluorescence_series.append(RoiResponseSeries(
                name=tbl.__name__,
                description=f"{tbl.__name__} - {attr_name}: {tbl.heading.attributes[attr_name].comment}",
                data=np.stack(roi_traces).T,
                rois=rt_region,
                unit="a.u.",
                timestamps=frame_timestamps,
            ))

        fl = Fluorescence(
            name=f"Fluorescence_{plane_name}",
            roi_response_series=fluorescence_series,
        )
        ophys_module.add(fl)

    return nwbfile


def write_nwb(nwbfile, fname, validate=False, check_read=True):
    """Export NWBFile

    Args:
        nwbfile (NWBFile): nwb file
        fname (str): Absolute path including `*.nwb` extension.
        validate (bool): If True, PyNWB will validate the produced NWB file.
        check_read (bool): If True, PyNWB will try to read the produced NWB file and
            ensure that it can be read.
    """
    with NWBHDF5IO(fname, "w") as io:
        io.write(nwbfile)

    if validate:
        import nwbinspector
        with NWBHDF5IO(fname, mode='r') as io:
            validation_status = pynwb.validate(io=io)
        logger.info(validation_status)
        for inspection_message in nwbinspector.inspect_all(path=fname):
            logger.info(inspection_message)

    if check_read:
        with NWBHDF5IO(fname, "r") as io:
            io.read()
    logger.info(f"File written successfully: {fname}")
