module Programs

export ConvexProgram
export initial, constraints!, objective!

abstract type ConvexProgram end

function initial end
function constraints! end
function objective! end

end
