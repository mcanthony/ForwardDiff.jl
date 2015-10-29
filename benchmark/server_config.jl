addprocs(3)

import BenchmarkTrackers, GitHub

@everywhere begin
    workspace = joinpath(homedir(), "benchmark_workspace", "ForwardDiff")
    if !(isdir(workspace))
        mkdir(workspace)
    end
    cd(workspace)
end

logger = BenchmarkTrackers.JSONLogger(workspace)
auth = GitHub.OAuth2(ENV["GITHUB_AUTH_TOKEN"]) # token granting repo permissions
secret = ENV["MY_SECRET"] # webhook secret
owner = "JuliaDiff"
repo = "ForwardDiff.jl"

server = BenchmarkTrackers.BenchmarkServer(logger, auth, secret, owner, repo)
BenchmarkTrackers.run(server, host=IPv4(127,0,0,1), port=2048)
