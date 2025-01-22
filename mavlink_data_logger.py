##########################################################################################
# This test script reads the GLOBAL_POSITION_INT (GPS) message from the MAVLink connection
# and prints it to the console. UDP-connection pre-configured to work with BlueBoat.

# IMPORTANT NOTE: We successfully tested this using GCS_Client_Link port and IP.
# This was set to 192.168.2.2:14550 at the time. It can be found in MavLink Endpoints menu,
# in pirate mode.


##########################################################################################

from pymavlink import mavutil

import pandas as pd
import os
import sys

def main():
      if len(sys.argv) < 2:
            print("Usage: python mavlink_data_logger.py <folder_name>")
            sys.exit(1)

      folder_name = sys.argv[1]
      log_dir = os.path.join('logs', folder_name)

      if not os.path.exists(log_dir):
            os.makedirs(log_dir)

      # Start a connection listening to a UDP port
      connection = mavutil.mavlink_connection('udpin:192.168.2.1:14550')

      # Wait for the first heartbeat
      # This sets the system and component ID of remote system for the link
      connection.wait_heartbeat()
      print(f'Heartbeat from the system (system {connection.target_system} component {connection.target_component})')

      # Create empty DataFrames for GPS and Odom messages
      gps_data = []
      odom_data = []

      try:
            while True:
                  msg_gps = connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
                  msg_gps = msg_gps.to_dict()
                  if msg_gps:
                        print("\n-------------------------- GPS ------------------------------")
                        print(msg_gps)
                        print("---------------------------------------------------------------\n")

                  msg_odom = connection.recv_match(type='RAW_IMU', blocking=True)
                  msg_odom = msg_odom.to_dict()
                  if msg_odom:
                        print("\n-------------------------- ODO ------------------------------")
                        print(msg_odom)
                        print("---------------------------------------------------------------\n")

                  # Append GPS data to the DataFrame
                  gps_data.append({
                        'lat': msg_gps['lat'],
                        'lon': msg_gps['lon'],
                        'alt': msg_gps['alt'],
                        'time_usec': msg_gps['time_boot_ms'],
                        'relative_alt': msg_gps["relative_alt"],
                        'vx': msg_gps['vx'],
                        'vy': msg_gps['vy'],
                        'vz': msg_gps['vy'],
                        'hdg': msg_gps['hdg'],
                  })

                  # Append Odom data to the DataFrame
                  odom_data.append({
                        'xacc': msg_odom['xacc'],
                        'yacc': msg_odom['yacc'],
                        'zacc': msg_odom['zacc'],
                        'xgyro': msg_odom['xgyro'],
                        'ygyro': msg_odom['ygyro'],
                        'zgyro': msg_odom['zgyro'],
                        'time_usec': msg_odom['time_usec']
                  })
      except KeyboardInterrupt:
            print("Logging interrupted by user")

      # Convert lists to DataFrames
      gps_df = pd.DataFrame(gps_data)
      odom_df = pd.DataFrame(odom_data)

      # Save DataFrames to CSV files with a timestamp
      timestamp = pd.Timestamp.now().strftime('%m%d_%H%M')
      gps_df.to_csv(os.path.join(log_dir, f'gps_data_{timestamp}.csv'), index=False)
      odom_df.to_csv(os.path.join(log_dir, f'odom_data_{timestamp}.csv'), index=False)

if __name__ == "__main__":
      main()