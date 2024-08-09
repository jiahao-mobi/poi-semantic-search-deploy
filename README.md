# poi-semantic-search-deploy

```
# building image
docker build -t semantic_search .

# run image
docker run -p 8080:80 semantic_search

# test user_query_tui_poi_search
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{
               "inputs": "Hotel by the water with bars",
               "topK": 5,
               "columns": [
                    "Expedia_Id",
                    "Name",
                    "Description"
               ]
          }' \
http://localhost:8080/user_query_tui_poi_search

# test user_query_tag_search
curl -X POST -H "Content-Type: application/json" -d '{                            
    "inputs": "Swire Hotels are my favorite.",
    "index": "ids-accommodation-brand-tag-vector",    
    "topK": 10
}' http://localhost:8080/user_query_tag_search

```