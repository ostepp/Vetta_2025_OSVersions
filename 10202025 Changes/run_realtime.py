import serial
import struct
# import time
import threading
import pandas as pd
import utils.read_packets as RP
import matplotlib.pyplot as plt
import numpy as np
from utils.config import PipelineConfig

from utils.stance import StanceAnalyzer
from utils.sensor import SensorData, input_df, _process_sensor_df

CRC_POLYNOMIAL = 0xAB

HEADER_FORMAT = '<BBBI'
FOOTER_FORMAT = '<B'
DATAPACKET_FORMAT = '<I9hBB'

HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # Calculate expected header size
FOOTER_SIZE = struct.calcsize(FOOTER_FORMAT)  # Calculate expected footer size

PORT = "COM3" # Com port of the device
BAUDRATE = 921600

serial_lock = threading.Lock()
serial_comm = None
# serial_comm = serial.Serial(PORT, BAUDRATE, timeout=1)
running = True

packet_analysis = True
analysis_window = 1000
Analysis_DF = pd.DataFrame(columns=['DeviceID', 'PacketID', 'Timestamp'])
duplicate_counter = 0
ooo_counter = 0
loss_counter = 0

# Based on configuration
accel_scale = 2.0 / 32768.0
gyro_scale = 250.0 / 32768.0
magneto_scale = 4.0 / 32768.0


csv_file = "IMU_data_RevB_v3.csv"


def handle_info_packet(packet_bytes):
    info_packet = RP.parse_info_packet(packet_bytes)

    # check crc here
    crc_caclulated = RP.calculate_crc8(packet_bytes[:HEADER_SIZE+info_packet['PayloadLen']])
    if(crc_caclulated != info_packet['CRC']):
        # Packet is corrupted. Discard the Data
        return

    # Display information
    print(f"Info Packet=>Time:{info_packet['Timestamp']}, Info:{info_packet['Info']}")

def handle_data_packet(packet_bytes):
    global Analysis_DF, duplicate_counter, ooo_counter, loss_counter
    # Parse the data
    data_packet = RP.parse_data_packet(packet_bytes)    

    # check crc here
    crc_caclulated = RP.calculate_crc8(packet_bytes[:HEADER_SIZE+data_packet['PayloadLen']])
    # assert crc_caclulated == data_packet['CRC'], "CRC Check Failed - data packet."
    if(crc_caclulated != data_packet['CRC']):
        # Packet is corrupted. Discard the Data
        return
    
    # Update the packet with meaningful values
    # Accelerometer conversion
    data_packet['AccelX'] =  data_packet['AccelX'] * accel_scale
    data_packet['AccelY'] =  data_packet['AccelY'] * accel_scale
    data_packet['AccelZ'] =  data_packet['AccelZ'] * accel_scale

    # Gyroscope conversion
    data_packet['GyroX'] =  data_packet['GyroX'] * gyro_scale
    data_packet['GyroY'] =  data_packet['GyroY'] * gyro_scale
    data_packet['GyroZ'] =  data_packet['GyroZ'] * gyro_scale

    # Magnetometer conversion
    data_packet['MagX'] =  data_packet['MagX'] * magneto_scale
    data_packet['MagY'] =  data_packet['MagY'] * magneto_scale
    data_packet['MagZ'] =  data_packet['MagZ'] * magneto_scale

    # Store the packet
    newdata = pd.DataFrame([data_packet])
    newdata.to_csv(csv_file, mode='a', index=False, header=False)

    if(packet_analysis):
        # Check for Duplicate packet
        duplicate = not Analysis_DF[(Analysis_DF['DeviceID'] == data_packet['DeviceID']) & (Analysis_DF['PacketID'] == data_packet['PacketID'])].empty
        if(duplicate):
            duplicate_counter+=1
            print(f"Duplicate Packet (DeviceID={data_packet['DeviceID']}, PacketID={data_packet['PacketID']}) | Counter = {duplicate_counter}")

        # # Check for OOO Packets
        # ooo_packets = not Analysis_DF[(Analysis_DF['DeviceID'] == data_packet['DeviceID']) & (data_packet['PacketID'] < Analysis_DF['PacketID'])].empty
        # if(ooo_packets):
        #     ooo_counter+=1
        #     print(f"Out of order Packet (DeviceID={data_packet['DeviceID']}, PacketID={data_packet['PacketID']}) | Counter = {ooo_counter}")

        # # Check for Missing packets
        # difference = data_packet['PacketID'] - Analysis_DF[(Analysis_DF['DeviceID'] == data_packet['DeviceID'])]['PacketID'].tail(1)
        # if not difference.empty and difference.iloc[0] > 1:
        #     print("Difference : ", difference.iloc[0])

        # Append new row after checking everything
        new_row = {'DeviceID':data_packet['DeviceID'], 'PacketID':data_packet['PacketID'], 'Timestamp':data_packet['Timestamp']}
        Analysis_DF = pd.concat([Analysis_DF, pd.DataFrame([new_row])], ignore_index=True)
        # delete older rows
        if len(Analysis_DF) > analysis_window:
            Analysis_DF = Analysis_DF.iloc[-analysis_window:].reset_index(drop=True)

    return data_packet

def parse_header(data):
    return struct.unpack(HEADER_FORMAT, data)

def read_serial():
    global running, serial_comm, ESPData, steps_df, vgrf_df, axes
    while running:
        # Read just the header
        header_bytes = serial_comm.read(HEADER_SIZE)
        if len(header_bytes) < HEADER_SIZE:
            continue

        try:
            packet_type, payload_len, device_id, timestamp = parse_header(header_bytes)
            remaining_bytes = serial_comm.read(payload_len + FOOTER_SIZE)
            # print('reading serial data')
            # Read the payload and footer based on packet_type
            if len(remaining_bytes) == payload_len + FOOTER_SIZE:
                if packet_type == 0x01:
                    handle_info_packet(header_bytes + remaining_bytes)
                elif packet_type == 0x02:
                    packet = handle_data_packet(header_bytes + remaining_bytes)
                    # TODO: here is where I expect we need to process packets in real time. These are global 
                    # variables, so they should still be present for exporting at the end. 
                    ESPData, steps_df, vgrf_df, axes = process_packet(packet, ESPData, steps_df, vgrf_df, axes)
                else:
                    # Received unknown packetS
                    pass
        
        except struct.error:
            print("Error unpacking data")

def send_data(message):
    with serial_lock:
        data_bytes = message.encode() + b'\n'
        serial_comm.write(data_bytes)
        print(f"Sent: {data_bytes}")


def process_packet(packet, stream_df, steps_df, vgrf_df, axes, verbose=False, plot=True):
    '''
    Process a single incoming packet and append to general and step dataframes.

    Args:
        packet: Dictionary containing packet data
        stream_df: DataFrame to append raw packet data
        steps_df: DataFrame to append detected steps
        vgrf_df: DataFrame to append vertical ground reaction forces
        axes: Array of Axes objects for plotting
        verbose: If True, print debug information
        plot: If True, update plots with new data

    Returns:
        Updated stream_df, steps_df, vgrf_df, and figure object
    '''

    # append packet to dataframe
    # if type(stream_df) is not pd.DataFrame:
    #     return stream_df, steps_df, vgrf_df, axes
    
    # if packet == None:
    #     return stream_df, steps_df, vgrf_df, axes

    if len(stream_df) == 0:
        time = 0.0
    else:
        time = (packet['Timestamp'] - stream_df['Timestamp'].min()) / 1000.0 # in seconds

    # print('Processing packets')
    stream_df.loc[len(stream_df)] = [packet['PacketType'], packet['PayloadLen'], packet['DeviceID'], 
                       packet['Timestamp'], packet['PacketID'], 
                    #    [packet['accel'][0], packet['accel'][1], packet['accel'][2]],
                       [packet['AccelX'], packet['AccelY'], packet['AccelZ']], # combine accel into list for norm calcs later
                    #    packet['GyroX'], packet['GyroY'], packet['GyroZ'],  # omit gyroscope
                    #    packet['MagX'], packet['MagY'], packet['MagZ'],  # omit magnetometer
                       packet['Flags'], packet['Battery'], packet['CRC'], time]
    

    # ensure enough data for analysis - set buffer size (est_samp_freq * search_seconds)
    est_samp_freq = 200
    search_seconds = 10
    start = len(stream_df) - est_samp_freq * search_seconds
    if verbose:
        print('Start index for analysis: ', start, '   Total Samples: ', len(stream_df))
    if start < 0 or len(stream_df) < est_samp_freq * search_seconds:
        # print("Not enough data for analysis")
        return stream_df, steps_df, vgrf_df, axes

    # dont analyze too often, only F frequency samples
    # when not within tolerance (tol) of frequency intervals, only append data and return
    # TODO: WILL NEED ADUSTMENT BASED ON SAMPLING RATE
    F = 4 # Hz
    times_to_analyze = np.linspace(0, 1, F+1)[:-1]
    # print(times_to_analyze)
    tol = 0.001
    rel_time = time - np.floor(time)
    diffs = [abs(rel_time - t) for t in times_to_analyze]
    if min(diffs) < tol:
        # print('Processing sensor data at time: ', time)
        pass
    else:
        return stream_df, steps_df, vgrf_df, axes


    # process sensor data (parse, resample, and filter)
    Sensors = input_df(stream_df.copy())
    if verbose:
        print(Sensors.keys(), Sensors['left'].shape, Sensors['right'].shape, Sensors['waist'].shape)

    ProcessedSensors = {}
    SensorNames = ['left', 'right', 'waist']
    for s in SensorNames:
        ProcessedSensors[s] = _process_sensor_df(Sensors[s])
    if any(ProcessedSensors[s] is None for s in SensorNames):
        print("Error processing sensor data")
        return stream_df, steps_df, vgrf_df, axes
    # print(ProcessedSensors)

    # set gait cycle identification parameters
    config = PipelineConfig(
        accel_peak_params={
            'height': 1.0,
            'prominence': 0.5,
            'width': 5.0,
            'distance': 10
        },
        jerk_peak_params={
            'height': 0.0,
            'prominence': 0.1
        },
        jerk_window_size=50,
        stance_matching_time_threshold=50,
        accel_filters=[],
        vgrf_filters=[],
        min_stance_size=60,
        max_stance_size=140
        )

    # identify steps and parse gait cycles
    # print('Searching for steps...')
    SA = StanceAnalyzer(config)
    left_strikes, left_stances = SA.extract_sensor_stances(
        ProcessedSensors['left']['accel_filtered'],
        ProcessedSensors['left']['accel'],
        ProcessedSensors['waist']['accel_filtered'],
    )
        
    right_strikes, right_stances = SA.extract_sensor_stances(
        ProcessedSensors['right']['accel_filtered'],    
        ProcessedSensors['right']['accel'],
        ProcessedSensors['waist']['accel_filtered'],
    )
    left_strike_times = [ProcessedSensors['left']['time'][i] for i in left_strikes]
    right_strike_times = [ProcessedSensors['right']['time'][i] for i in right_strikes]

    if time > 5: # and float(str(time).split('.')[1]) < 0.1: 
        print(f"{time} s,  {len(stream_df)} streaming frames,  {len(left_stances)} Left steps,  {len(right_stances)} Right steps")
        # print(f"Left stances: {len(left_stances)}  Right stances: {len(right_stances)}")


    # log output in steps df
    new_l_inds = []
    new_r_inds = []
    if len(left_strikes) > 0:
        if left_strikes[-1] not in steps_df['End_Frame'].values:
            if len(left_strikes) > 2:
                start_frame = left_strikes[-2]
                end_frame = left_strikes[-1]
                side = 'left'
                id = 'L' + str(len(steps_df[steps_df['Side'] == 'left']))
                waist_data = list(left_stances[-1]) 
                new_l_inds.append(len(steps_df))
                steps_df.loc[len(steps_df)] = [time, side, start_frame, end_frame, id] + waist_data

    if len(right_strikes) > 0:
        if right_strikes[-1] not in steps_df['End_Frame'].values:
            if len(right_strikes) > 2:
                start_frame = right_strikes[-2]
                end_frame = right_strikes[-1]
                side = 'right'
                id = 'R' + str(len(steps_df[steps_df['Side'] == 'right']))
                waist_data = list(right_stances[-1]) 
                new_r_inds.append(len(steps_df))
                steps_df.loc[len(steps_df)] = [time, side, start_frame, end_frame, id] + waist_data


    # send to model for prediction
    for l_ind in new_l_inds:
        waist_acc_cols = [col for col in steps_df.columns if 'waist_accel' in col]
        l_waist_acc = steps_df[waist_acc_cols].iloc[l_ind]
        l_waist_acc.columns = range(0, 100)
        from utils.predict import predict_stance
        l_output = predict_stance(l_waist_acc)

        # save output to vgrfs_df
        peak_vgrf = l_output[:50].max()
        vgrf_df.loc[len(vgrf_df)] = [steps_df['ID'].iloc[l_ind], peak_vgrf] + list(l_output)

    for r_ind in new_r_inds:
        waist_acc_cols = [col for col in steps_df.columns if 'waist_accel' in col]
        r_waist_acc = steps_df[waist_acc_cols].iloc[r_ind]
        r_waist_acc.columns = range(0, 100)
        from utils.predict import predict_stance
        r_output = predict_stance(r_waist_acc)

        # save output to vgrfs_df
        peak_vgrf = r_output[:50].max()
        vgrf_df.loc[len(vgrf_df)] = [steps_df['ID'].iloc[r_ind], peak_vgrf] + list(r_output)


    # plot results
    if time > 10 and plot:

        ax1 = axes[0]  # Top subplot
        ax2 = axes[1]  # Middle subplot
        ax3 = axes[2]  # Bottom subplot
        A1 = 0.5
        A2 = 0.3

        # realtime plot of input acceleration data
        ax1.plot(ProcessedSensors['left']['time'], ProcessedSensors['left']['accel_filtered'], 
                 color='C0', label='Left Accel Filtered', alpha=A1)
        ax1.plot(ProcessedSensors['right']['time'], ProcessedSensors['right']['accel_filtered'], 
                 color='C1', label='Right Accel Filtered', alpha=A1)
        ax1.plot(ProcessedSensors['waist']['time'], ProcessedSensors['waist']['accel_filtered'], 
                 color='C2', label='Waist Accel Filtered', alpha=A1)
        for x in left_strike_times:
            ax1.axvline(x=x, color='C0', linestyle='--', alpha=A1)
        for x in right_strike_times:
            ax1.axvline(x=x, color='C1', linestyle='--', alpha=A1)
        # ax1.legend(fontsize='small')
        ax1.set_title('Streaming Acceleration Data')
        ax1.set_ylabel('Accel (g)')
        ax1.set_xlabel('Time (s)')

        # extracted & parsed waist acceleration to send to model
        waist_acc_cols = [col for col in steps_df.columns if 'waist_accel' in col]
        l_waist_acc = steps_df[waist_acc_cols].iloc[new_l_inds]
        l_waist_acc.columns = range(0, 100)
        r_waist_acc = steps_df[waist_acc_cols].iloc[new_r_inds]
        r_waist_acc.columns = range(0, 100)
        ax2.plot(l_waist_acc.T, color='C0', alpha=A2, lw=2)
        ax2.plot(r_waist_acc.T, color='C1', alpha=A2, lw=2)
        ax2.set_title('Step-Parsed Waist Accelerations')
        ax2.set_ylabel('Accel (g)')
        ax2.set_xlabel('% Gait Cycle')

        # model predicted vGRF
        vgrf_cols = [col for col in vgrf_df.columns if 'vGRF_' in col]
        l_vgrf = vgrf_df[vgrf_cols].iloc[new_l_inds]
        l_vgrf.columns = range(0, 100)
        r_vgrf = vgrf_df[vgrf_cols].iloc[new_r_inds]
        r_vgrf.columns = range(0, 100)
        ax3.plot(l_vgrf.T, color='C0', alpha=A2, lw=2)
        ax3.plot(r_vgrf.T, color='C1', alpha=A2, lw=2)
        ax3.set_title('Step-Predicted vGRFs')
        ax3.set_ylabel('vGRF (BW)')
        ax3.set_xlabel('% Gait Cycle')

        plt.show(block=False)
        plt.tight_layout()
        plt.savefig('acc_steps_preds.png')

        
    return stream_df, steps_df, vgrf_df, axes


def main(run):
    """Main function to run real-time data processing from serial port or CSV file.
    args:
        run: 
            'serial' to read sensor data in real time from serial port, 
            'csv' to read from CSV file and play through the data as if real-time.
    """
    global serial_comm, running, ESPData, steps_df, vgrf_df, axes
    try:
        while True:

            

            # initialize dataframes to store results
            # w_acc_cols = [f'waist_accel_{i}' for i in range(0, 100)]
            # steps_df = pd.DataFrame(columns=['Timestamp', 'Side', 'Start_Frame', 'End_Frame', 'ID'] + w_acc_cols) 
            # vgrf_cols = [f'vGRF_{i}' for i in range(0, 100)]
            # vgrf_df = pd.DataFrame(columns=['ID','PeakvGRF'] + vgrf_cols)
            # ESPData = pd.DataFrame(columns=['PacketType', 'PayloadLen', 'DeviceID', 
            #                             'Timestamp', 'PacketID', 'accel', 
            #                             'Flags', 'Battery', 'CRC', 'time'])
            
            # create figure and subplots for showing data
            # fig, axes = plt.subplots(nrows=3, ncols=1)
            # fig.set_figheight(10) 
            # fig.set_figwidth(8)


            if run == 'serial':
                # running in serial mode
                # global serial_comm, running, ESPData, steps_df, vgrf_df
                

                if serial_comm == None:
                    serial_comm = serial.Serial(PORT, BAUDRATE, timeout=1)
                    print(f"Connecting to {PORT} at {BAUDRATE} baud...")

                    w_acc_cols = [f'waist_accel_{i}' for i in range(0, 100)]
                    steps_df = pd.DataFrame(columns=['Timestamp', 'Side', 'Start_Frame', 'End_Frame', 'ID'] + w_acc_cols) 
                    vgrf_cols = [f'vGRF_{i}' for i in range(0, 100)]
                    vgrf_df = pd.DataFrame(columns=['ID','PeakvGRF'] + vgrf_cols)
                    ESPData = pd.DataFrame(columns=['PacketType', 'PayloadLen', 'DeviceID', 
                                            'Timestamp', 'PacketID', 'accel', 
                                            'Flags', 'Battery', 'CRC', 'time'])
                    fig, axes = plt.subplots(nrows=3, ncols=1)
                    fig.set_figheight(10) 
                    fig.set_figwidth(8)

                else:
                    print('serial port already connected, continuing...')

                reader_thread = threading.Thread(target=read_serial, daemon=True)
                reader_thread.start()

                user_input = input("")
                # if user_input.lower() == 'exit':
                #     break
                # print('right before sending data')
                send_data(user_input)
                # print('running: ', running)

                # plt.tight_layout()
                # plt.show()
                # fig.savefig('acc_steps_preds.png')


            elif run == 'csv':
                # load data to simulate real-time processing loop
                fn = 'IMU_data_ricky_walk.csv'
                data = pd.read_csv(fn)
                print(f'loading csv data from: {fn}')
                print(data.head())

                w_acc_cols = [f'waist_accel_{i}' for i in range(0, 100)]
                steps_df = pd.DataFrame(columns=['Timestamp', 'Side', 'Start_Frame', 'End_Frame', 'ID'] + w_acc_cols) 
                vgrf_cols = [f'vGRF_{i}' for i in range(0, 100)]
                vgrf_df = pd.DataFrame(columns=['ID','PeakvGRF'] + vgrf_cols)

                ESPData = pd.DataFrame(columns=['PacketType', 'PayloadLen', 'DeviceID', 
                                        'Timestamp', 'PacketID', 'accel', 
                                        'Flags', 'Battery', 'CRC', 'time'])
                
                # create figure and subplots for showing data
                fig, axes = plt.subplots(nrows=3, ncols=1)
                fig.set_figheight(10) 
                fig.set_figwidth(8)

                # simulate real-time data reception
                print('\nStarting simulation...')
                for index, row in data.iterrows():
                    packet = row.to_dict()

                    if len(packet) == 0:
                        continue

                    print(index)
                    print('Packet received:   ', packet)
                    ESPData, steps_df, vgrf_df, axes = process_packet(packet, ESPData, steps_df, vgrf_df, axes)

                    if index > 14000:
                        print("\nEnding simulation.")
                        print('Streaming DF: \n', ESPData.tail())
                        print('Steps DF: \n', steps_df.tail())
                        print('vGRF DF: \n', vgrf_df.tail())

                        ax1 = axes[0] 
                        window = 10
                        if max(ESPData['time']) > window:
                            ax1.set_xlim([max(ESPData['time']) - window, max(ESPData['time'])])

                        plt.tight_layout()
                        plt.show()
                        fig.savefig('acc_steps_preds.png')

                        print('Ending Simulation early')
                        break

                break # exit csv running when all rows ran through
                    
            else:
                raise ValueError("Invalid run mode. Use 'serial' or 'csv'.")
            
            
    except KeyboardInterrupt:
        pass

    finally:
    # if not running:
        if run == 'serial':
            running = False
            serial_comm.close()
            print("Serial port closed.")

        csv_file = 'ESP_stream_data.csv'
        steps_file = 'Parsed_steps_data.csv'
        vgrf_file = 'Predicted_vGRF_data.csv'
        ESPData.to_csv(csv_file, index=False)
        steps_df.to_csv(steps_file, index=False)
        vgrf_df.to_csv(vgrf_file, index=False)
        print(f"Data saved to {csv_file} and {steps_file} and {vgrf_file}")


if __name__ == "__main__":
    main(run='serial') # change to serial to run with sensors connected to the serial port