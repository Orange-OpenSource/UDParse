# Software Name: UDParse
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 Orange
# SPDX-License-Identifier: Mozilla Public License 2.0
#
# This software is distributed under the MPL-2.0 license.
# the text of which is available at https://www.mozilla.org/en-US/MPL/2.0/
# or see the "LICENSE" file for more details.
#
# Author: Johannes HEINECKE <johannes(dot)heinecke(at)orange(dot)com> et al.

VERSION_MAJOR = 2
VERSION_MINOR = 3
VERSION_PATCH = 0

def getVersion():
    return "%s.%s.%s" % (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)

VERSION = getVersion()
