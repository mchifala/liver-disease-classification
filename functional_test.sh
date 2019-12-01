test -e ssshtest || curl -O https://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest
. ssshtest

run test_overall python rest_client.py localhost test_one 0 1 25 .5 .1 100 20 12 5.9 1.6 1
assert_no_stderr
assert_exit_code 0

run test_overall python rest_client.py localhost test_one 0 1 25 .5 .1 100 20 12 5.9 1.6 bad_input
assert_no_stderr
assert_exit_code 0