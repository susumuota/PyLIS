# -*- coding: utf-8 -*-

# Copyright 2019 Susumu OTA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# see https://wp.bmemo.pw/1304


import sys
import csv
import argparse
from operator import add
from functools import reduce
from datetime import datetime, timedelta, timezone

import boto3


def get_region_names(region_name):
    ec2 = boto3.client('ec2', region_name=region_name)
    region_names = [ region['RegionName'] for region in ec2.describe_regions()['Regions'] ]
    return region_names

def get_spot_price_history(region_name, instance_types, os_types, history_time):
    ec2 = boto3.client('ec2', region_name=region_name)
    spot_price_history = ec2.describe_spot_price_history(
        InstanceTypes=instance_types,
        ProductDescriptions=os_types,
        StartTime=history_time.isoformat()
    )
    return spot_price_history.get('SpotPriceHistory')

def get_sorted_spot_price_list(region_name, instance_types, os_types, history_time):
    regions = get_region_names(region_name)
    histories = [ get_spot_price_history(region, instance_types, os_types, history_time) for region in regions ]
    prices = sorted(reduce(add, histories), key=lambda h: -1.0*float(h['SpotPrice']))
    # {'AvailabilityZone': 'ap-south-1b', 'InstanceType': 'm5d.24xlarge', 'ProductDescription': 'Linux/UNIX', 'SpotPrice': '1.460400', 'Timestamp': datetime.datetime(2019, 2, 11, 6, 58, 45, tzinfo=tzutc())}
    return [ [ p['SpotPrice'], p['InstanceType'], p['AvailabilityZone'], p['ProductDescription'] ] for p in prices ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region_name', type=str, default='us-east-2', help='region name to query.')
    parser.add_argument('--instance_types', type=str, default='p2.xlarge,c5.xlarge', help='instance types. comma separated.')
    parser.add_argument('--os_types', type=str, default='Linux/UNIX', help='OS types. comma separated.')
    parser.add_argument('--delimiter', type=str, default='\t', help='CSV output delimiter.')
    args = parser.parse_args()
    history_time = (datetime.now(timezone.utc) + timedelta(hours=0)) # TODO: hours
    out = csv.writer(sys.stdout, delimiter=args.delimiter)
    for p in get_sorted_spot_price_list(args.region_name, args.instance_types.split(','), args.os_types.split(','), history_time):
        out.writerow(p)
