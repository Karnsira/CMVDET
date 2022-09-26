def gson = GsonTools.getInstance(true)

def json = new File("<Your Output Json path>").text

// Read the annotations
def type = new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>() {}.getType()
def deserializedAnnotations = gson.fromJson(json, type)

addObjects(deserializedAnnotations)   