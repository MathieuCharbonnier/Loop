import opensim as osim

def solveMocoInverse():

    inverse = osim.MocoInverse()
    #model_file = "data/gait2392_millard2012muscle.osim"
    model_file= "data/BothLegs.osim"
    motion_file = "data/BothLegsWalk.mot"

    # Load and simplify the model
    modelProcessor = osim.ModelProcessor(model_file)


    # Replace muscles for remaining coordinates
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())

    # Keep muscles simple
    modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.2))
    modelProcessor.append(osim.ModOpAddReserves(10.0))

    inverse.setModel(modelProcessor)

    # Load motion data
    table = osim.TimeSeriesTable(motion_file)
    print("Motion file time range:", table.getIndependentColumn()[0], "to", table.getIndependentColumn()[-1])
    print("Motion file columns:", table.getColumnLabels())

    inverse.setKinematics(osim.TableProcessor(motion_file))
    inverse.set_kinematics_allow_extra_columns(True)
    inverse.set_mesh_interval(0.02)

    # Solve
    solution = inverse.solve()
    solution.getMocoSolution().write('MocoInverse_solution.sto')

    # Report
    model = modelProcessor.process()
    report = osim.report.Report(model, 'MocoInverse_solution.sto', bilateral=True)
    report.generate()

solveMocoInverse()
