import copyreg
try:
    from keras_core.src.utils.tracking import TrackedList, TrackedDict,  TrackedSet
    for name in ['TrackedList', 'TrackedDict', 'TrackedSet']:
        if locals()[name] is not None:
            def pickle_tracked(tracked):
                return globals()[name], (tracked, tracked.tracker)


            copyreg.pickle(globals()[name], pickle_tracked)
except ImportError:
    TrackedList = None
    TrackedDict = None
    TrackedSet = None
