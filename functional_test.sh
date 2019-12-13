test -e ssshtest || curl -O https://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest
. ssshtest

# Start up rest_server using nohup
- nohup python rest_server.py &

# Test rest_client and rest_server with proper input
run test_client python rest_client.py localhost test_one 0 1 25 .5 .1 100 20 12 5.9 1.6 1
assert_no_stderr
assert_exit_code 0

# Test rest_client and rest_server with bad input
run test_client2 python rest_client.py localhost test_one 0 1 25 .5 .1 100 20 12 5.9 1.6 bad_input
assert_no_stderr
assert_exit_code 0

# Test the selection of a model via pipeline and k-fold cross validation
run test_model_select python select_model.py config_dummy.json
assert_no_stderr
assert_exit_code 0